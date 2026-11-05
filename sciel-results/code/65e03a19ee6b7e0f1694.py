"""
Strategy: Basin Hopping with gradient-based local minimization.
Instead of simulated annealing with random perturbations, we use basin hopping:
1. Start from diverse initial structures (icosahedral, random, etc.)
2. Locally minimize each using L-BFGS-B (fast, gradient-based)
3. Apply a random displacement (kick) to escape the local minimum
4. Re-minimize, and accept if energy decreased
5. Track the best structure found across all restarts

This is fundamentally different from SA: we exploit the gradient information
heavily via L-BFGS-B for fast convergence to local minima, then use random
kicks to hop between basins. The kicks are larger than SA steps because we
don't need to stay in the neighborhood.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

def _dzugutov_energy_and_grad(pos):
    """
    Compute energy and gradient of the Dzugutov potential.
    pos: (n, 3) array
    Returns: energy (float), grad (n, 3) array
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = ~np.eye(n, dtype=bool)
    dist_safe = np.where(mask, dist, 1.0)
    
    V = np.zeros((n, n))
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    if np.any(mask_a):
        r_a = dist_safe[mask_a]
        term1 = A * (r_a ** (-m) - B)
        term2 = np.exp(c / (r_a - a))
        V_vals = term1 * term2
        dterm1 = -m * (r_a ** (-m - 1))
        dterm2 = -c / (r_a - a)**2
        dV_vals = (dterm1 * term2 + term1 * term2 * dterm2)
        V[mask_a] = V_vals
        dV[mask_a] = dV_vals
    
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    if np.any(mask_b):
        r_b = dist_safe[mask_b]
        V_vals_b = B * np.exp(d / (r_b - b))
        dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
        V[mask_b] = V_vals_b
        dV[mask_b] = dV_vals_b
    
    energy = 0.5 * np.sum(V)
    
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _check_min_dist(pos, min_d=0.5):
    """Check if all interatomic distances are >= min_d."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _generate_icosahedron_shell(n_atoms, radius=1.0):
    """Generate points based on icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    rng = np.random.default_rng(42)
    r_shell = radius * (n_atoms / 12.0) ** (1.0/3.0)
    
    points = [np.zeros(3)]
    for v in verts:
        points.append(r_shell * v)
    
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5:
            points.append(p)
    
    return np.array(points[:n_atoms])

def _generate_random_compact(n_atoms, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    r_max = (n_atoms * 0.15) ** (1/3)
    points = []
    for _ in range(n_atoms * 10):
        if len(points) >= n_atoms:
            break
        p = rng.uniform(-r_max, r_max, 3)
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) > 0.5:
                points.append(p)
    if len(points) < n_atoms:
        # Fallback: grid
        side = int(np.ceil(n_atoms ** (1/3)))
        k = 0
        for i in range(side):
            for j in range(side):
                for l in range(side):
                    if k < n_atoms:
                        points.append(np.array([i, j, l]) * 0.6)
                        k += 1
    return np.array(points[:n_atoms])

def _local_minimize(pos, max_iter=500):
    """
    Locally minimize using L-BFGS-B.
    Returns (energy, positions).
    """
    n = len(pos)
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(n, 3)
        e, g = _dzugutov_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(fun, x0, jac=True, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-8})
        pos_opt = result.x.reshape(n, 3)
        e_opt = result.fun
        return e_opt, pos_opt
    except Exception:
        return _dzugutov_energy_and_grad(pos)[0], pos

def _basin_hopping(pos_init, n_hops=20, kick_scale=0.5, seed=0, time_limit=60.0):
    """
    Basin hopping: minimize, kick, minimize, accept if lower energy.
    """
    rng = np.random.default_rng(seed)
    n = len(pos_init)
    
    # First minimize
    best_energy, best_pos = _local_minimize(pos_init)
    current_energy, current_pos = best_energy, best_pos
    
    start_time = time.time()
    
    for hop in range(n_hops):
        if time.time() - start_time > time_limit:
            break
        
        # Random kick
        kick = rng.normal(0, kick_scale, (n, 3))
        kicked_pos = current_pos + kick
        
        # Check min distance after kick
        if not _check_min_dist(kicked_pos):
            # Try smaller kick
            for attempt in range(5):
                kick = rng.normal(0, kick_scale * 0.5, (n, 3))
                kicked_pos = current_pos + kick
                if _check_min_dist(kicked_pos):
                    break
        
        if not _check_min_dist(kicked_pos):
            continue
        
        # Minimize the kicked structure
        e_new, pos_new = _local_minimize(kicked_pos)
        
        # Accept if lower energy (Metropolis-like with T=0 for simplicity)
        if e_new < current_energy:
            current_energy = e_new
            current_pos = pos_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
    
    return best_pos, best_energy

def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Icosahedral seed
    try:
        ico_pos = _generate_icosahedron_shell(n, radius=1.0)
        if _check_min_dist(ico_pos):
            candidates.append(ico_pos)
    except:
        pass
    
    # 2. Random compact seeds with different seeds
    for seed in [1, 2, 3, 4]:
        try:
            rand_pos = _generate_random_compact(n, seed=seed)
            if _check_min_dist(rand_pos):
                candidates.append(rand_pos)
        except:
            pass
    
    # 3. Gaussian cluster
    try:
        rng = np.random.default_rng(99)
        gauss_pos = rng.normal(0, 0.8, (n, 3))
        if _check_min_dist(gauss_pos):
            candidates.append(gauss_pos)
    except:
        pass
    
    if not candidates:
        # Fallback: grid
        pos = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        k = 0
        for i in range(side):
            for j in range(side):
                for l in range(side):
                    if k < n:
                        pos[k] = [i, j, l] * 0.6
                        k += 1
        return pos[:n]
    
    # Sort candidates by initial energy
    scored_candidates = []
    for pos in candidates:
        try:
            e, _ = _dzugutov_energy_and_grad(pos)
            scored_candidates.append((e, pos))
        except:
            pass
    
    if not scored_candidates:
        return candidates[0]
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_overall_energy = np.inf
    best_overall_pos = None
    
    # Allocate time across candidates
    n_candidates = len(scored_candidates)
    time_per_candidate = 250.0 / n_candidates
    
    for idx, (e_init, pos_init) in enumerate(scored_candidates):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        alloc_time = min(time_per_candidate, remaining_time)
        
        if alloc_time < 5.0:
            break
        
        # Run basin hopping
        try:
            pos_bh, e_bh = _basin_hopping(
                pos_init, 
                n_hops=30, 
                kick_scale=0.5, 
                seed=idx * 1000,
                time_limit=alloc_time
            )
            
            if e_bh < best_overall_energy:
                best_overall_energy = e_bh
                best_overall_pos = pos_bh.copy()
        except Exception:
            pass
        
        # Polish with relax if time permits
        if best_overall_pos is not None and time.time() < deadline - 5.0:
            try:
                e_polished, pos_polished = relax(best_overall_pos, 'dzugutov')
                if _check_min_dist(pos_polished) and e_polished < best_overall_energy:
                    best_overall_energy = e_polished
                    best_overall_pos = pos_polished.copy()
            except:
                pass
    
    if best_overall_pos is None:
        return scored_candidates[0][1]
        
    return best_overall_pos
# EVOLVE-BLOCK-END
