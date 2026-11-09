"""Morse cluster global minima: Adaptive multi-scale basin hopping with gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 315.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0


def _morse_energy_grad(rho, coords):
    """Energy and gradient for Morse potential."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10, np.zeros_like(coords)
    
    e = np.exp(rho * (1.0 - d))
    energy = np.sum(e * (e - 2.0)) / 2.0
    
    # Gradient: dE/dr_ij = 2*e*(e-1)*(-rho) for each pair
    # But we need dE/dx_i, so use chain rule
    grad = np.zeros_like(coords)
    for i in range(n):
        for j in range(i+1, n):
            rij = coords[i] - coords[j]
            dist = d[i, j]
            if dist < 1e-10:
                continue
            eij = np.exp(rho * (1.0 - dist))
            de = 2.0 * eij * (eij - 1.0) * (-rho)
            grad[i] += de * rij / dist
            grad[j] -= de * rij / dist
    
    return energy, grad


def _validate_and_fix(coords, max_iter=50):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords.copy()
    coords = coords.copy()
    for _ in range(max_iter):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= 0.5:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            diff = np.array([0.01, 0.01, 0.01])
            dist = np.linalg.norm(diff)
        diff /= dist
        push = (0.5 - dist) * 0.5 + 0.01
        coords[i] += push * diff
        coords[j] -= push * diff
    return coords


def _relax_safe(coords, rho):
    """Relax with fallback validation."""
    try:
        e, x = relax(coords, 'morse', rho)
        if e < 1e8:
            return e, x
        else:
            x_fixed = _validate_and_fix(x)
            e2, x2 = relax(x_fixed, 'morse', rho)
            if e2 < 1e8:
                return e2, x2
            return 1e10, x_fixed
    except Exception:
        try:
            x_fixed = _validate_and_fix(coords)
            e, x = relax(x_fixed, 'morse', rho)
            if e < 1e8:
                return e, x
        except Exception:
            pass
        return 1e10, coords


def _gradient_polish(coords, rho, max_iter=50):
    """Use scipy minimize with analytic gradient for fine-tuning."""
    def fun_and_grad(flat):
        c = flat.reshape(-1, 3)
        e, g = _morse_energy_grad(rho, c)
        return e, g.reshape(-1)
    
    flat = coords.reshape(-1)
    res = minimize(fun_and_grad, flat, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'gtol': 1e-8})
    if res.success and res.fun < 1e8:
        return res.fun, res.x.reshape(-1, 3)
    return _morse_energy(rho, coords), coords


def _adaptive_perturb(coords, rng, base_scale, n_perturbed=None):
    """Perturb a random subset with adaptive scale."""
    n = len(coords)
    if n_perturbed is None:
        n_perturbed = max(1, int(n * (0.05 + 0.3 * rng.random())))
    idx = rng.choice(n, size=min(n_perturbed, n), replace=False)
    noise = rng.normal(0.0, base_scale, size=(len(idx), 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _basin_hopping_chain(rho, x0, rng, max_iter=300, temp_start=0.15, temp_end=0.005):
    """
    Basin hopping with adaptive perturbation scale and gradient polish.
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    e_rel, x_rel = _relax_safe(x0, rho)
    if e_rel < 1e8:
        cur_x = x_rel
        cur_e = e_rel
        if cur_e < best_e:
            best_e = cur_e
            best_x = x_rel.copy()
    
    # Gradient polish on initial
    if _remaining() > 5:
        e_gp, x_gp = _gradient_polish(cur_x, rho, max_iter=30)
        if e_gp < 1e8 and e_gp < cur_e:
            cur_x = x_gp
            cur_e = e_gp
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_gp.copy()
    
    # Adaptive temperature and scale
    scale_min = 0.02
    scale_max = 0.3
    stagnation_count = 0
    max_stagnation = 20
    
    for i in range(max_iter):
        if _remaining() < 3.0:
            break
        
        t = i / max_iter
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Adaptive perturbation scale: increase if stuck
        if stagnation_count > max_stagnation // 2:
            base_scale = scale_max
        elif stagnation_count > max_stagnation // 4:
            base_scale = (scale_min + scale_max) / 2
        else:
            base_scale = scale_min + (scale_max - scale_min) * (1 - t) * 0.5
        
        # Perturb
        x_new = _adaptive_perturb(cur_x, rng, base_scale)
        
        # Relax
        e_rel, x_rel = _relax_safe(x_new, rho)
        
        if e_rel > 1e8:
            continue
        
        # Gradient polish on new point
        if _remaining() > 5 and (i % 3 == 0 or e_rel < cur_e):
            e_gp, x_gp = _gradient_polish(x_rel, rho, max_iter=20)
            if e_gp < 1e8:
                e_rel = e_gp
                x_rel = x_gp
        
        dE = e_rel - cur_e
        # Metropolis acceptance with occasional forced acceptance for exploration
        accept = False
        if dE < 0:
            accept = True
            stagnation_count = 0
        elif rng.random() < np.exp(-dE / max(temp, 1e-10)):
            accept = True
            stagnation_count += 1
        else:
            stagnation_count += 1
        
        if accept:
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
                stagnation_count = 0
        else:
            stagnation_count += 1
    
    # Final gradient polish on best
    if _remaining() > 3:
        e_gp, x_gp = _gradient_polish(best_x, rho, max_iter=50)
        if e_gp < 1e8 and e_gp < best_e:
            best_e = e_gp
            best_x = x_gp.copy()
    
    return best_x, best_e


def _generate_structured_seeds(n, rng):
    """Generate diverse structured initial configurations."""
    seeds = []
    
    # 1. FCC-like packing
    a = 1.0
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    fcc_seed = pts[order[:n]].copy()
    fcc_seed += rng.normal(0, 0.03, fcc_seed.shape)
    seeds.append(fcc_seed)

    # 2. HCP-like packing
    a_hcp = 1.0
    c_over_a = 1.633  # ideal HCP ratio
    pts_hcp = []
    for layer in range(-3, 4):
        z = layer * c_over_a * a_hcp
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * a_hcp + (j % 2) * a_hcp * 0.5
                y = j * a_hcp * np.sqrt(3) / 2
                pts_hcp.append([x, y, z])
    pts_hcp = np.array(pts_hcp)
    pts_hcp -= pts_hcp.mean(axis=0)
    dists_hcp = np.linalg.norm(pts_hcp, axis=1)
    order_hcp = np.argsort(dists_hcp)
    hcp_seed = pts_hcp[order_hcp[:n]].copy()
    hcp_seed += rng.normal(0, 0.03, hcp_seed.shape)
    seeds.append(hcp_seed)

    # 3. Icosahedral core-shell
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    if n <= 12:
        ico_seed = ico[:n].copy()
    else:
        # Add concentric shells
        remaining = n - 12
        shell_radii = [1.8, 2.5, 3.2]
        shell_pts = []
        for r in shell_radii:
            if remaining <= 0:
                break
            # Fibonacci sphere for this shell
            golden = (1 + np.sqrt(5)) / 2
            n_on_shell = min(remaining, int(4 * np.pi * r**2 / 1.2))
            n_on_shell = max(1, min(n_on_shell, remaining))
            for i in range(n_on_shell):
                z = 1 - 2 * (i + 0.5) / n_on_shell
                rr = np.sqrt(max(0, 1 - z * z))
                theta = 2 * np.pi * i / golden
                shell_pts.append([rr * r * np.cos(theta), rr * r * np.sin(theta), z * r])
            remaining -= n_on_shell
        
        # Fill remaining with random interior
        if remaining > 0:
            for _ in range(remaining):
                p = rng.uniform(-0.5, 0.5, 3)
                if np.linalg.norm(p) < 0.5:
                    shell_pts.append(p)
        
        shell_pts = np.array(shell_pts)
        ico_seed = np.vstack([ico, shell_pts])[:n]
    
    ico_seed += rng.normal(0, 0.03, ico_seed.shape)
    seeds.append(ico_seed)

    # 4. Spherical shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.6
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.03, shell_seed.shape)
    seeds.append(shell_seed)

    # 5. Two-shell structure
    if n > 10:
        outer_n = max(10, int(n * 0.55))
        inner_n = n - outer_n
        outer_r = 1.3
        inner_r = 0.6
        
        outer_pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            outer_pts.append([rr * np.cos(theta), rr * np.sin(theta), z])
        outer_pts = np.array(outer_pts) * outer_r
        
        inner_pts = rng.uniform(-inner_r, inner_r, size=(inner_n, 3))
        two_shell = np.vstack([outer_pts, inner_pts])
        two_shell += rng.normal(0, 0.03, two_shell.shape)
        seeds.append(two_shell)

    # 6. Random compact
    radius = n ** (1/3) * 0.7
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    return seeds


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 7919 + int(rho) * 104729)
    
    # Generate structured seeds
    seeds = _generate_structured_seeds(n, rng)
    
    # Relax all seeds to get baselines
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        e, x = _relax_safe(seed, rho)
        if e < 1e8:
            # Gradient polish
            e_gp, x_gp = _gradient_polish(x, rho, max_iter=20)
            if e_gp < 1e8:
                e = e_gp
                x = x_gp
            if e < best_e:
                best_e = e
                best_x = x.copy()
    
    if best_x is None:
        # Fallback: return first seed
        return seeds[0]
    
    # Parallel basin hopping with 4 chains
    num_chains = 4
    chain_starts = []
    
    # Use best seed and a few others
    chain_starts.append(best_x.copy())
    for s in seeds[:num_chains-1]:
        chain_starts.append(s.copy())
    
    # Ensure we have enough chains
    while len(chain_starts) < num_chains:
        r = n ** (1/3) * 0.7
        chain_starts.append(rng.uniform(-r, r, size=(n, 3)))
    
    results = []
    
    def _run_chain(args):
        rho_val, x0, thread_rng = args
        return _basin_hopping_chain(rho_val, x0, thread_rng, max_iter=250, temp_start=0.15, temp_end=0.005)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chains) as executor:
        futures = []
        for i in range(num_chains):
            thread_rng = np.random.default_rng(rng.integers(0, 2**32))
            futures.append(executor.submit(_run_chain, (rho, chain_starts[i].copy(), thread_rng)))
        
        for future in concurrent.futures.as_completed(futures, timeout=_remaining()):
            try:
                x_bh, e_bh = future.result(timeout=2.0)
                if e_bh < 1e8:
                    results.append((x_bh, e_bh))
            except Exception:
                continue
    
    # Combine results
    for x_bh, e_bh in results:
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh.copy()
    
    # Final polish: multiple small perturbations + gradient
    if _remaining() > 8.0:
        for _ in range(10):
            if _remaining() < 3.0:
                break
            x_pert = _adaptive_perturb(best_x, rng, 0.03, n_perturbed=max(1, int(n*0.1)))
            e, x = _relax_safe(x_pert, rho)
            if e < 1e8:
                e_gp, x_gp = _gradient_polish(x, rho, max_iter=30)
                if e_gp < 1e8 and e_gp < best_e:
                    best_e = e_gp
                    best_x = x_gp.copy()
    
    # Final gradient polish
    if _remaining() > 3.0:
        e_final, x_final = _gradient_polish(best_x, rho, max_iter=50)
        if e_final < 1e8 and e_final < best_e:
            best_e = e_final
            best_x = x_final.copy()
    
    # Validate
    e_check = _morse_energy(rho, best_x)
    if e_check < 1e8:
        return best_x
    
    # Fallback
    return best_x
# EVOLVE-BLOCK-END
