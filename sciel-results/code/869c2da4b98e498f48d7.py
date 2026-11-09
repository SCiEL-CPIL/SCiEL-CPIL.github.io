"""Morse cluster global minima: Parallel Basin Hopping with diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import concurrent.futures
import threading

_START = time.time()
_BUDGET = 310.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET

_lock = threading.Lock()
_shared_best_e = np.inf
_shared_best_x = None


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
    coords = coords.copy()
    for _ in range(200):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= 0.5:
            break
        # Find all pairs below threshold and push them apart
        below = d < 0.5
        np.fill_diagonal(below, False)
        rows, cols = np.where(below)
        for i, j in zip(rows, cols):
            if i >= j:
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


def _perturb(coords, rng, scale):
    """Random displacement of a random subset of atoms."""
    n = len(coords)
    k = max(1, int(n * (0.1 + 0.2 * rng.random())))
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.normal(0.0, scale, size=(k, 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _basin_hopping_chain(rho, x0, rng, max_iter=200, temp_start=0.3, temp_end=0.01):
    """
    Basin hopping with cooling temperature.
    Returns (best_x, best_e).
    """
    global _shared_best_e, _shared_best_x
    
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    e_rel, x_rel = relax(x0, 'morse', rho)
    if e_rel < 1e8:
        cur_x = x_rel
        cur_e = e_rel
        if cur_e < best_e:
            best_e = cur_e
            best_x = x_rel.copy()

    for i in range(max_iter):
        if _remaining() < 3.0:
            break
        
        # Cooling schedule
        t = i / max_iter
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Perturb with varying scale
        scale = 0.15 + 0.15 * np.sin(np.pi * t)
        x_new = _perturb(cur_x, rng, scale)
        
        # Relax
        e_rel, x_rel = relax(x_new, 'morse', rho)
        
        if e_rel > 1e8:
            # Try to fix and retry once
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel > 1e8:
                continue

        dE = e_rel - cur_e
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
                
        # Update shared best
        with _lock:
            if best_e < _shared_best_e:
                _shared_best_e = best_e
                _shared_best_x = best_x.copy()
                
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

    # 2. Icosahedral structure
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
        # Build multi-shell icosahedral structure
        inner_n = n - 12
        inner_r = 0.55
        inner_pts = []
        # Use Fibonacci sphere for inner points
        golden = (1 + np.sqrt(5)) / 2
        for i in range(inner_n):
            z = 1 - 2 * (i + 0.5) / inner_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            inner_pts.append([rr * np.cos(theta) * inner_r, rr * np.sin(theta) * inner_r, z * inner_r])
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.03, ico_seed.shape)
    seeds.append(ico_seed)

    # 3. Random Compact
    radius = n ** (1/3) * 0.7
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 4. Spherical Shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.5
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.03, shell_seed.shape)
    seeds.append(shell_seed)

    # 5. Two-Shell Structure
    if n > 10:
        outer_n = max(10, int(n * 0.65))
        inner_n = n - outer_n
        outer_r = 1.1
        inner_r = 0.45
        
        outer_pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            outer_pts.append([rr * np.cos(theta), rr * np.sin(theta), z])
        outer_pts = np.array(outer_pts) * outer_r
        
        inner_pts = []
        for i in range(inner_n):
            z = 1 - 2 * (i + 0.5) / inner_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            inner_pts.append([rr * np.cos(theta) * inner_r, rr * np.sin(theta) * inner_r, z * inner_r])
        inner_pts = np.array(inner_pts)
        
        two_shell = np.vstack([outer_pts, inner_pts])
        two_shell += rng.normal(0, 0.03, two_shell.shape)
        seeds.append(two_shell)

    # 6. HCP-like packing
    a = 1.0
    hcp_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            layer = (i + j) % 2
            x = i * a + (0.5 if layer else 0)
            y = j * a * np.sqrt(3) / 2
            z = layer * a * 0.5
            hcp_pts.append([x, y, z])
    hcp_pts = np.array(hcp_pts)
    hcp_pts -= hcp_pts.mean(axis=0)
    dists = np.linalg.norm(hcp_pts, axis=1)
    order = np.argsort(dists)
    hcp_seed = hcp_pts[order[:n]].copy()
    hcp_seed += rng.normal(0, 0.03, hcp_seed.shape)
    seeds.append(hcp_seed)

    # 7. BCC-like packing
    a = 1.0
    bcc_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                bcc_pts.append([i*a, j*a, k*a])
                bcc_pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    bcc_pts = np.array(bcc_pts)
    bcc_pts -= bcc_pts.mean(axis=0)
    dists = np.linalg.norm(bcc_pts, axis=1)
    order = np.argsort(dists)
    bcc_seed = bcc_pts[order[:n]].copy()
    bcc_seed += rng.normal(0, 0.03, bcc_seed.shape)
    seeds.append(bcc_seed)

    return seeds


def _worker(rho, n, x0, rng_seed):
    """Worker function for parallel basin hopping."""
    rng = np.random.default_rng(rng_seed)
    x_best, e_best = _basin_hopping_chain(rho, x0, rng, max_iter=150, temp_start=0.3, temp_end=0.01)
    return x_best, e_best


def construct(instance):
    global _shared_best_e, _shared_best_x
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Reset shared state
    _shared_best_e = np.inf
    _shared_best_x = None
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    # Relax all seeds first to get a baseline
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        e, x = relax(seed, 'morse', rho)
        if e < 1e8:
            if e < best_e:
                best_e = e
                best_x = x.copy()
        else:
            # Try to fix
            x_fixed = _validate_and_fix(x)
            e, x = relax(x_fixed, 'morse', rho)
            if e < 1e8:
                if e < best_e:
                    best_e = e
                    best_x = x.copy()

    if best_x is None:
        # Fallback
        return seeds[0]

    # Update shared best
    _shared_best_e = best_e
    _shared_best_x = best_x.copy()

    # Parallel Basin Hopping with 4 threads
    num_chains = len(seeds) + 4  # Add extra random starts
    chain_starts = seeds[:]
    for _ in range(num_chains - len(seeds)):
        r = n ** (1/3) * 0.7
        chain_starts.append(rng.uniform(-r, r, size=(n, 3)))
    
    # Pre-relax chain starts
    for i in range(len(chain_starts)):
        e, x = relax(chain_starts[i], 'morse', rho)
        if e < 1e8:
            chain_starts[i] = x
        else:
            chain_starts[i] = _validate_and_fix(x)

    # Run parallel basin hopping with 4 threads
    max_workers = 4
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, x0 in enumerate(chain_starts):
            if _remaining() < 10.0:
                break
            seed_val = n * 1000 + int(rho) + i * 7919
            futures.append(executor.submit(_worker, rho, n, x0, seed_val))
        
        # Wait for all futures with timeout
        for future in concurrent.futures.as_completed(futures, timeout=max(1.0, _remaining() - 5.0)):
            try:
                x_best, e_best = future.result(timeout=1.0)
                with _lock:
                    if e_best < _shared_best_e:
                        _shared_best_e = e_best
                        _shared_best_x = x_best.copy()
            except Exception:
                pass

    # Use the best found
    if _shared_best_x is not None and _shared_best_e < 1e8:
        best_x = _shared_best_x.copy()
        best_e = _shared_best_e
    elif best_x is not None:
        pass  # Keep original best
    else:
        return seeds[0]

    # Final Polish
    if _remaining() > 8.0:
        # Do a few more relaxations on the best found
        for _ in range(10):
            if _remaining() < 3.0:
                break
            # Small perturbation and relax
            x_pert = _perturb(best_x, rng, 0.05)
            e, x = relax(x_pert, 'morse', rho)
            if e < 1e8 and e < best_e:
                best_e = e
                best_x = x.copy()
        
        # One final relax
        _, best_x = relax(best_x, 'morse', rho)
        
        # Validate
        if _morse_energy(rho, best_x) < 1e8:
            return best_x

    # Final validation
    if _morse_energy(rho, best_x) < 1e8:
        return best_x
    
    # If invalid, try to fix
    best_x = _validate_and_fix(best_x)
    e, best_x = relax(best_x, 'morse', rho)
    if _morse_energy(rho, best_x) < 1e8:
        return best_x
    
    # Last resort
    return _validate_and_fix(seeds[0])
# EVOLVE-BLOCK-END
