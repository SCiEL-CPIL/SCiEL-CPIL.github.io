"""
Strategy: Basin Hopping with L-BFGS-B local minimization and polytetrahedral seeding.
Instead of simulated annealing with random perturbations, we use basin hopping:
1. Start from diverse initial structures (icosahedral shells, random compact)
2. Locally minimize each using L-BFGS-B (fast, gradient-based)
3. Apply a random displacement (kick) to escape the local minimum
4. Re-minimize, and accept if energy decreased (Metropolis criterion)
5. Track the best structure found across all restarts

This is fundamentally different from SA: we exploit the gradient information
heavily via L-BFGS-B for fast convergence to local minima, then use larger
kicks to jump between basins. This should find lower-energy local minima
more efficiently than SA's small random steps.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

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
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    V = np.zeros((n, n))
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    r_a = dist_safe[mask_a]
    if len(r_a) > 0:
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
    r_b = dist_safe[mask_b]
    if len(r_b) > 0:
        V_vals_b = B * np.exp(d / (r_b - b))
        dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
        V[mask_b] = V_vals_b
        dV[mask_b] = dV_vals_b
    
    energy = 0.5 * np.sum(V)
    
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _check_min_dist(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist

def _generate_icosahedron_shell(n_atoms, radius=1.0):
    """
    Generate points based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Scale to fit n atoms roughly
    r_shell = radius * (n_atoms / 12.0) ** (1.0/3.0)
    
    points = [np.zeros(3)]
    for v in verts:
        points.append(r_shell * v)
    
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5:
            points.append(p)
    
    return np.array(points[:n_atoms])

def _generate_random_compact(n_atoms, seed=0):
    """Generate random compact structure."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n_atoms, 3))
    # Normalize to roughly same volume
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pos = pos / norms * (n_atoms ** (1/3)) * 0.8
    return pos

def _local_minimize(pos, deadline):
    """
    Perform local minimization using L-BFGS-B.
    Returns (energy, position) or (inf, pos) if fails.
    """
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(pos)
        return e, g.ravel()
    
    try:
        result = minimize(
            objective,
            pos.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < np.inf:
            pos_opt = result.x.reshape(-1, 3)
            e_opt = result.fun
            return e_opt, pos_opt
        else:
            return np.inf, pos
    except Exception:
        return np.inf, pos

def _basin_hopping(pos_init, n_kicks, kick_sigma, deadline, seed=0):
    """
    Perform basin hopping.
    """
    rng = np.random.default_rng(seed)
    
    # First local minimize
    e_curr, pos_curr = _local_minimize(pos_init, deadline)
    if e_curr >= np.inf:
        e_curr, pos_curr = _dzugutov_energy_and_grad(pos_init)
    
    e_best = e_curr
    pos_best = pos_curr.copy()
    
    for i in range(n_kicks):
        if time.time() > deadline:
            break
        
        # Apply random kick
        noise = rng.normal(0, kick_sigma, pos_curr.shape)
        pos_kicked = pos_curr + noise
        
        # Check min distance
        if not _check_min_dist(pos_kicked):
            # Try to fix by scaling
            pos_kicked = pos_kicked * 1.1
            if not _check_min_dist(pos_kicked):
                continue
        
        # Local minimize the kicked structure
        e_new, pos_new = _local_minimize(pos_kicked, deadline)
        if e_new >= np.inf:
            continue
        
        # Metropolis criterion
        delta_e = e_new - e_curr
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.05):
            e_curr = e_new
            pos_curr = pos_new
            if e_curr < e_best:
                e_best = e_curr
                pos_best = pos_curr.copy()
        
        # Adaptive kick size
        kick_sigma = np.clip(kick_sigma * (0.99 if delta_e < 0 else 1.01), 0.05, 1.0)
    
    return pos_best, e_best

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
    
    # 2. Random compact seeds with different scales
    for scale in [0.7, 0.8, 0.9, 1.0]:
        rng = np.random.default_rng()
        pos = rng.uniform(-1, 1, (n, 3))
        norms = np.linalg.norm(pos, axis=1, keepdims=True)
        norms[norms == 0] = 1
        pos = pos / norms * (n ** (1/3)) * scale
        if _check_min_dist(pos):
            candidates.append(pos)
    
    # 3. Layered structure
    rng = np.random.default_rng(123)
    side = int(np.ceil(n ** (1/3)))
    pos = np.zeros((n, 3))
    k = 0
    for i in range(side):
        for j in range(side):
            for l in range(side):
                if k < n:
                    pos[k] = [i, j, l] * 0.6
                    k += 1
    if _check_min_dist(pos):
        candidates.append(pos)
    
    if not candidates:
        # Fallback: grid placement
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
            if e < np.inf:
                scored_candidates.append((e, pos))
        except:
            pass
    
    if not scored_candidates:
        return candidates[0]
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_overall_energy = np.inf
    best_overall_pos = None
    
    # Run basin hopping on top candidates
    # Allocate time based on number of candidates
    n_candidates = min(len(scored_candidates), 4)
    time_per_candidate = (deadline - time.time()) / n_candidates
    
    for idx in range(n_candidates):
        if time.time() > deadline:
            break
        
        e_init, pos_init = scored_candidates[idx]
        
        # Run basin hopping with adaptive kick size
        # Start with moderate kick, let it adapt
        n_kicks = 30
        kick_sigma = 0.15
        
        t_start = time.time()
        pos_bh, e_bh = _basin_hopping(pos_init, n_kicks, kick_sigma, deadline, seed=idx)
        t_elapsed = time.time() - t_start
        
        if e_bh < best_overall_energy:
            best_overall_energy = e_bh
            best_overall_pos = pos_bh
        
        # If we have time, do a second pass with smaller kicks for polishing
        if time.time() < deadline - 30:
            n_kicks2 = 20
            kick_sigma2 = 0.05
            pos_bh2, e_bh2 = _basin_hopping(pos_bh, n_kicks2, kick_sigma2, deadline, seed=idx + 100)
            if e_bh2 < best_overall_energy:
                best_overall_energy = e_bh2
                best_overall_pos = pos_bh2
    
    # Final polish with relax if we have time
    if best_overall_pos is not None and time.time() < deadline - 10:
        try:
            e_polished, pos_polished = relax(best_overall_pos, 'dzugutov')
            if _check_min_dist(pos_polished) and e_polished < best_overall_energy:
                best_overall_energy = e_polished
                best_overall_pos = pos_polished
        except:
            pass
    
    if best_overall_pos is None:
        return scored_candidates[0][1]
    
    return best_overall_pos
# EVOLVE-BLOCK-END
