"""Morse cluster global minima: Basin Hopping with gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 295.0  # Safety margin before 300s kill
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


def _validate_and_fix(coords):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        for _ in range(200):
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
            if dist < 1e-6:
                diff = np.array([0.01, 0.01, 0.01])
                dist = np.linalg.norm(diff)
            diff /= dist
            coords[i] += 0.005 * diff
            coords[j] -= 0.005 * diff
    return coords


def _perturb(coords, rng, scale, frac=0.25):
    """Random displacement of a random subset of atoms."""
    n = len(coords)
    k = max(1, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.normal(0.0, scale, size=(k, 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _gradient_polish(rho, x, max_iter=50, tol=1e-8):
    """Final high-precision polish using gradient descent with adaptive step size."""
    x = x.copy()
    e = _morse_energy(rho, x)
    if e > 1e8:
        return x, e
    
    step = 0.01
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        try:
            e_grad, grad = morse_energy_gradient(x, rho)
        except Exception:
            break
        
        if e_grad > 1e8:
            break
        
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Adaptive step size
        x_try = x - step * grad
        e_try = _morse_energy(rho, x_try)
        
        if e_try < e:
            x = x_try
            e = e_try
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    
    return x, e


def _basin_hopping_chain(rho, x0, rng, time_budget=10.0):
    """
    Basin hopping with adaptive perturbation and cooling.
    Returns (best_x, best_e).
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        if e_rel < 1e8:
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        else:
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel < 1e8:
                cur_x = x_rel
                cur_e = e_rel
                if cur_e < best_e:
                    best_e = cur_e
                    best_x = x_rel.copy()
    except Exception:
        pass
    
    # Estimate cost per iteration
    t0 = time.time()
    n_iter = 0
    temp = 0.1
    scale_base = 0.15
    frac_base = 0.25
    stuck_count = 0
    
    while _remaining() > 2.0 and (time.time() - t0) < time_budget:
        # Adaptive perturbation: increase when stuck
        if stuck_count > 5:
            scale = scale_base * (1.0 + 0.5 * np.log1p(stuck_count / 5.0))
            frac = min(0.5, frac_base + 0.05 * (stuck_count // 5))
        else:
            scale = scale_base
            frac = frac_base
        
        x_new = _perturb(cur_x, rng, scale, frac)
        
        try:
            e_rel, x_rel = relax(x_new, 'morse', rho)
        except Exception:
            continue
        
        if e_rel > 1e8:
            x_fixed = _validate_and_fix(x_rel)
            try:
                e_rel, x_rel = relax(x_fixed, 'morse', rho)
            except Exception:
                continue
            if e_rel > 1e8:
                continue
        
        dE = e_rel - cur_e
        # Metropolis acceptance with geometric cooling
        accept = dE < 0 or rng.random() < np.exp(-dE / temp)
        
        if accept:
            cur_x = x_rel
            cur_e = e_rel
            stuck_count = 0
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        else:
            stuck_count += 1
        
        # Geometric cooling
        temp *= 0.98
        n_iter += 1
    
    return best_x, best_e


def _generate_seeds(n, rng):
    """Generate a diverse set of initial configurations."""
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

    # 2. BCC-like packing
    pts_bcc = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts_bcc.append([i*1.0, j*1.0, k*1.0])
                pts_bcc.append([(i+0.5), (j+0.5), (k+0.5)])
    pts_bcc = np.array(pts_bcc)
    pts_bcc -= pts_bcc.mean(axis=0)
    dists_bcc = np.linalg.norm(pts_bcc, axis=1)
    order_bcc = np.argsort(dists_bcc)
    bcc_seed = pts_bcc[order_bcc[:n]].copy()
    bcc_seed += rng.normal(0, 0.03, bcc_seed.shape)
    seeds.append(bcc_seed)

    # 3. Icosahedral shell
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
        inner_n = n - 12
        inner_r = 0.5
        inner_pts = []
        for _ in range(inner_n):
            while True:
                p = rng.uniform(-inner_r, inner_r, 3)
                if np.linalg.norm(p) < inner_r:
                    break
            inner_pts.append(p)
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.03, ico_seed.shape)
    seeds.append(ico_seed)

    # 4. Random Compact
    radius = n ** (1/3) * 0.7
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 5. Spherical Shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.45
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.03, shell_seed.shape)
    seeds.append(shell_seed)

    # 6. Two-Shell Structure
    if n > 10:
        outer_n = max(10, int(n * 0.55))
        inner_n = n - outer_n
        outer_r = 1.1
        inner_r = 0.4
        
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

    # 7. Cubic shell
    if n > 8:
        cube_pts = []
        side = max(2, int(round(n ** (1/3))))
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    cube_pts.append([i - side/2 + 0.5, j - side/2 + 0.5, k - side/2 + 0.5])
        cube_pts = np.array(cube_pts[:n])
        cube_pts += rng.normal(0, 0.03, cube_pts.shape)
        seeds.append(cube_pts)

    return seeds


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    # Relax all seeds first to get a baseline
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        try:
            e, x = relax(seed, 'morse', rho)
            if e < 1e8:
                if e < best_e:
                    best_e = e
                    best_x = x.copy()
            else:
                x_fixed = _validate_and_fix(x)
                e, x = relax(x_fixed, 'morse', rho)
                if e < 1e8:
                    if e < best_e:
                        best_e = e
                        best_x = x.copy()
        except Exception:
            continue

    if best_x is None:
        return seeds[0]

    # Basin Hopping: run multiple chains
    # Allocate time: 70% to BH, 20% to additional seeds, 10% to polish
    bh_time = _remaining() * 0.70
    num_chains = len(seeds) + 3  # Extra random starts
    
    chain_starts = seeds[:]
    for _ in range(num_chains - len(seeds)):
        r = n ** (1/3) * 0.7
        chain_starts.append(rng.uniform(-r, r, size=(n, 3)))
    
    time_per_chain = bh_time / num_chains
    
    for i in range(num_chains):
        if _remaining() < 10.0:
            break
        if time_per_chain < 2.0:
            break
            
        x0 = chain_starts[i].copy()
        x_bh, e_bh = _basin_hopping_chain(rho, x0, rng, time_budget=time_per_chain)
        
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh.copy()

    # Final gradient polish
    if _remaining() > 5.0:
        best_x, best_e = _gradient_polish(rho, best_x, max_iter=100, tol=1e-9)
        
        # One more small perturbation + relax + polish cycle
        if _remaining() > 3.0:
            x_pert = _perturb(best_x, rng, 0.03, 0.1)
            try:
                e_pert, x_pert = relax(x_pert, 'morse', rho)
                if e_pert < 1e8:
                    x_pert, e_pert = _gradient_polish(rho, x_pert, max_iter=50, tol=1e-9)
                    if e_pert < best_e:
                        best_e = e_pert
                        best_x = x_pert.copy()
            except Exception:
                pass
    
    # Final validation
    final_e = _morse_energy(rho, best_x)
    if final_e > 1e8:
        # Last resort: return a valid random config
        r = n ** (1/3) * 0.5
        fallback = rng.uniform(-r, r, size=(n, 3))
        fallback = _validate_and_fix(fallback)
        return fallback
    
    return best_x
# EVOLVE-BLOCK-END
