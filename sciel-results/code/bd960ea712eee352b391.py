"""Morse cluster global minima: Direct gradient-based basin hopping with diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 310.0  # Safety margin before 330s kill
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

def _fix_overlaps(coords, min_dist=0.5, max_iter=500):
    """Push apart any atoms closer than min_dist."""
    coords = coords.copy()
    n = len(coords)
    for _ in range(max_iter):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= min_dist:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            coords[i] += 0.01 * np.array([1, 0, 0])
            continue
        diff /= dist
        push = (min_dist - dist) * 0.5 + 0.01
        coords[i] += push * diff
        coords[j] -= push * diff
    return coords

def _local_minimize(rho, x0, max_iter=300):
    """
    Perform local minimization using L-BFGS-B with morse_energy_gradient.
    Returns (energy, coords).
    """
    n = len(x0)
    x0 = _fix_overlaps(x0)
    x0_flat = x0.flatten()
    
    def func_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        e, g = morse_energy_gradient(x, rho)
        return e, g.flatten()
        
    try:
        res = minimize(func_and_grad, x0_flat, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 50})
        x_opt = res.x.reshape(n, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt < 1e8:
            return e_opt, x_opt
    except Exception:
        pass
    
    # Fallback to relax
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        if e_rel < 1e8:
            return e_rel, x_rel
    except Exception:
        pass
    
    return 1e10, x0

def _generate_fcc_seeds(n, rng, count=3):
    """Generate FCC-like seeds with perturbations and rotations."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.8)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_fcc = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_fcc @ R.T
        noisy = rotated + rng.normal(0, 0.12, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_hcp_seeds(n, rng, count=2):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.8)) + 2
    for k in range(-limit, limit + 1):
        z = k * c
        for i in range(-limit, limit + 1):
            for j in range(-limit, limit + 1):
                if k % 2 == 0:
                    pts.append([i*a, j*a, z])
                    pts.append([(i+0.5)*a, (j+0.5)*a, z])
                else:
                    pts.append([(i+0.5)*a, j*a, z])
                    pts.append([i*a, (j+0.5)*a, z])
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_hcp = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_hcp @ R.T
        noisy = rotated + rng.normal(0, 0.12, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_icosahedral_seeds(n, rng, count=2):
    """Generate icosahedral-like seeds."""
    seeds = []
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    radii = [1.0, 1.7, 2.4, 3.1, 3.8, 4.5]
    for r in radii:
        for p in base:
            pts.append(p * r)
            pts.append(-p * r)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_icosa = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_icosa @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_random_compact_seeds(n, rng, count=3):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        box_size = (n * 0.35) ** (1/3) * 2.8
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        seeds.append(_fix_overlaps(pts))
    return seeds

def _basin_hopping_chain(rho, x0, rng, duration, seed_id):
    """
    Run a single basin hopping chain for a specified duration.
    """
    # Initial relax
    e_rel, x_rel = _local_minimize(rho, x0)
    if e_rel > 1e8:
        return x0, 1e10

    cur_e = e_rel
    best_e = e_rel
    best_x = x_rel.copy()
    cur_x = x_rel.copy()
    
    temp = 0.35
    t_start = time.time()
    n = len(x0)
    accepted = 0
    total = 0
    
    while _remaining() > 1.5 and (time.time() - t_start) < duration:
        # Adaptive shake scale
        shake_scale = 0.12 + 0.2 * np.sqrt(temp)
        
        # Perturb a random subset of atoms
        n_perturb = max(1, int(n * (0.1 + 0.1 * np.sqrt(temp))))
        idx = rng.choice(n, size=n_perturb, replace=False)
        noise = rng.normal(0, shake_scale, size=(n_perturb, 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        # Relax
        e_new, x_new_rel = _local_minimize(rho, x_new)
        
        if e_new > 1e8:
            continue

        total += 1
        dE = e_new - cur_e
        if dE < 0 or (temp > 0.01 and rng.random() < np.exp(-dE / temp)):
            cur_e = e_new
            cur_x = x_new_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_new_rel.copy()
                accepted += 1
        
        # Cool down
        temp *= 0.97
        # Reheat if stuck
        if total > 40 and accepted == 0:
            temp = max(temp, 0.25)
            accepted = 0
            total = 0
    
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    fcc_seeds = _generate_fcc_seeds(n, rng, count=3)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=2)
    icosa_seeds = _generate_icosahedral_seeds(n, rng, count=2)
    random_seeds = _generate_random_compact_seeds(n, rng, count=3)
    
    # Select 5 diverse starting points
    chains_starts = [fcc_seeds[0], hcp_seeds[0], icosa_seeds[0], random_seeds[0], random_seeds[1]]
    
    total_time = _remaining()
    polish_time = max(10.0, total_time * 0.12)
    search_time = total_time - polish_time
    time_per_chain = search_time / 5.0
    
    best_x_global = None
    best_e_global = np.inf
    
    def run_chain(args):
        x0, t_dur, seed_id = args
        x_res, e_res = _basin_hopping_chain(rho, x0, np.random.default_rng(seed_id), t_dur, seed_id)
        return x_res, e_res

    args = []
    for i, start in enumerate(chains_starts):
        args.append((start, time_per_chain, n * 100 + i))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_chain, arg) for arg in args]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                x_res, e_res = future.result()
                if e_res < best_e_global:
                    best_e_global = e_res
                    best_x_global = x_res
            except Exception:
                pass
                
    if best_x_global is None:
        return _fix_overlaps(chains_starts[0])
        
    # Polish phase: additional basin hopping on the best result
    if _remaining() > polish_time:
        rng_polish = np.random.default_rng(42 + n)
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time, 999)
        
    # Final check
    e_final = _morse_energy(rho, best_x_global)
    if e_final < 1e8:
        return best_x_global
    else:
        best_x_global = _fix_overlaps(best_x_global)
        e_final = _morse_energy(rho, best_x_global)
        if e_final < 1e8:
            return best_x_global
        return _fix_overlaps(chains_starts[0])

# EVOLVE-BLOCK-END
