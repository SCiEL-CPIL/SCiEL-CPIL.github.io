"""Morse cluster global minima: Parallel Adaptive Basin Hopping."""
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

def _morse_energy_and_grad(rho, coords):
    """Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10, np.zeros_like(coords)
    
    # Use the provided helper for accuracy and speed
    # Note: The helper might not include the penalty logic, so we check min d first.
    e_calc, g_calc = morse_energy_gradient(coords, rho)
    return e_calc, g_calc

def _fix_overlaps(x, n_iter=50):
    """Attempt to fix overlapping atoms by pushing them apart."""
    x = x.copy()
    for _ in range(n_iter):
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= 0.5:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = x[i] - x[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            x[i] += 0.01
            continue
        diff /= dist
        # Push apart
        x[i] += 0.05 * diff
        x[j] -= 0.05 * diff
    return x

def _local_minimize(rho, x0):
    """
    Perform local minimization using Steepest Descent.
    Returns (energy, coords).
    """
    n = len(x0)
    x = x0.copy()
    
    # Check for overlaps first
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        x = _fix_overlaps(x)

    e, g = _morse_energy_and_grad(rho, x)
    if e > 1e8:
        return e, x
        
    alpha = 0.01
    max_iter = 100
    for _ in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-4:
            break
        
        # Normalize gradient
        g_dir = g / g_norm
        
        # Try step
        x_new = x - alpha * g_dir
        e_new, g_new = _morse_energy_and_grad(rho, x_new)
        
        if e_new < e:
            x = x_new
            e = e_new
            g = g_new
            alpha *= 1.05 # Increase step slightly if successful
            if alpha > 0.1:
                alpha = 0.1
        else:
            alpha *= 0.5 # Reduce step if unsuccessful
            if alpha < 1e-5:
                break
                
    return e, x

def _generate_fcc_seeds(n, rng, count=5):
    """Generate multiple FCC-like seeds with slight perturbations and rotations."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
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
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _generate_random_compact_seeds(n, rng, count=5):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        # Heuristic box size: volume per atom ~ 1/2 for density 2
        # r ~ (n / (4/3 pi * 2))^(1/3)
        # Simplified: box_side ~ n^(1/3) * 1.5
        box_size = (n ** (1/3)) * 1.5
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        # Center
        pts -= pts.mean(axis=0)
        seeds.append(pts)
    return seeds

def _generate_hcp_seeds(n, rng, count=2):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # HCP structure: layers stacked ABC
                # Simplified: just use a close-packed lattice approximation
                # Actually, let's just use a dense random grid and sort by distance, similar to FCC but different geometry
                # Or generate specific HCP coordinates
                pass
                
    # Better HCP generation
    # Generate points in a hexagonal lattice
    for layer in range(int(np.ceil(n**(1/3))) + 2):
        z = layer * 0.816 * a # HCP layer spacing
        for i in range(-limit, limit + 1):
            for j in range(-limit, limit + 1):
                x = (i + 0.5 * (j % 2)) * 1.0 * a
                y = j * 0.866 * a
                pts.append([x, y, z])
                # Add shifted layer for B position
                pts.append([x + 0.5, y + 0.288, z + 0.408])

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
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _generate_icosahedral_seeds(n, rng, count=2):
    """Generate Icosahedral seeds."""
    seeds = []
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of icosahedron
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate a cluster of icosahedra
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.5)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # Place an icosahedron at each grid point
                center = np.array([i, j, k])
                scaled_verts = verts * 0.8 # Scale down to avoid overlap between icosahedra
                pts.extend(center + scaled_verts)
                
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_ico = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_ico @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(noisy)
        
    return seeds

def _basin_hopping_chain(rho, x0, rng, duration, seed_id):
    """
    Run a single basin hopping chain for a specified duration.
    """
    best_x = x0.copy()
    cur_x = x0.copy()
    
    # Initial relax
    e_rel, x_rel = _local_minimize(rho, x0)
    if e_rel > 1e8:
        return best_x, 1e10

    cur_e = e_rel
    best_e = e_rel
    best_x = x_rel.copy()
    
    temp = 0.5
    t_start = time.time()
    n = len(x0)
    
    while _remaining() > 1.0 and (time.time() - t_start) < duration:
        shake_scale = 0.2 + 0.3 * np.sqrt(temp)
        
        idx = rng.choice(n, size=max(1, int(n*0.2)), replace=False)
        noise = rng.normal(0, shake_scale, size=(len(idx), 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        # Relax
        e_rel, x_rel = _local_minimize(rho, x_new)
        
        if e_rel > 1e8:
            continue

        dE = e_rel - cur_e
        # Use local rng for Metropolis step to ensure reproducibility per chain
        if dE < 0 or (temp > 0 and rng.random() < np.exp(-dE / temp)):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        
        temp *= 0.95
        
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    fcc_seeds = _generate_fcc_seeds(n, rng, count=2)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=2)
    ico_seeds = _generate_icosahedral_seeds(n, rng, count=2)
    random_seeds = _generate_random_compact_seeds(n, rng, count=2)
    
    chains_starts = []
    chains_starts.append(fcc_seeds[0])
    chains_starts.append(hcp_seeds[0])
    chains_starts.append(ico_seeds[0])
    chains_starts.append(random_seeds[0])
    
    total_time = _remaining()
    # Reserve time for polish
    polish_time = max(10.0, total_time * 0.15)
    search_time = total_time - polish_time
    time_per_chain = search_time / 4.0
    
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
            except Exception as e:
                pass
                
    if best_x_global is None:
        # Fallback
        best_x_global = chains_starts[0]
        best_e_global = _morse_energy(rho, best_x_global)
        
    # Polish phase
    if _remaining() > polish_time and best_x_global is not None:
        rng_polish = np.random.default_rng(42)
        # Run a shorter, cooler BH chain on the best candidate
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time, 999)
        
    # Final exact relaxation to ensure we are at the minimum
    if best_x_global is not None:
        e_final, x_final = _local_minimize(rho, best_x_global)
        if e_final < 1e8:
            best_x_global = x_final
            best_e_global = e_final
            
    e_final = _morse_energy(rho, best_x_global)
    if e_final < 1e8:
        return best_x_global
    else:
        # Emergency fix
        best_x_global = _fix_overlaps(best_x_global)
        return best_x_global

# EVOLVE-BLOCK-END
