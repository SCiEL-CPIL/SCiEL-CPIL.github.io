"""
Strategy: Basin Hopping with L-BFGS-B Local Minimization.
The Dzugutov potential is smooth and differentiable. Instead of Simulated Annealing,
we use Basin Hopping:
1. Generate diverse initial structures (Icosahedral, Random, Shell).
2. For each structure, perform a cycle of:
   a. Local minimization using L-BFGS-B (fast, gradient-based).
   b. Random perturbation (kick) to escape the local minimum.
   c. Accept the new structure if the energy after re-minimization is lower.
3. Track the global best structure across all cycles and restarts.

This approach leverages the gradient to converge quickly to local minima,
making the "kick" steps more meaningful as they start from a well-defined
basin center rather than an arbitrary point in the basin.
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

def _dzugutov_energy(pos):
    """
    Compute energy of the Dzugutov potential.
    pos: (n, 3) array
    """
    n = len(pos)
    if n < 2:
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = ~np.eye(n, dtype=bool)
    dist_safe = np.where(mask, dist, 1.0)
    
    energy = 0.0
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    r_a = dist_safe[mask_a]
    if len(r_a) > 0:
        term1 = A * (r_a ** (-m) - B)
        term2 = np.exp(c / (r_a - a))
        energy += np.sum(term1 * term2)
        
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    r_b = dist_safe[mask_b]
    if len(r_b) > 0:
        V_vals_b = B * np.exp(d / (r_b - b))
        energy += np.sum(V_vals_b)
        
    return 0.5 * energy

def _dzugutov_gradient(pos):
    """
    Compute gradient of the Dzugutov potential.
    """
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
        
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(n, dtype=bool)
    dist_safe = np.where(mask, dist, 1.0)
    
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    r_a = dist_safe[mask_a]
    if len(r_a) > 0:
        term1 = A * (r_a ** (-m) - B)
        term2 = np.exp(c / (r_a - a))
        # dV/dr
        dterm1 = -m * (r_a ** (-m - 1))
        dterm2 = -c / (r_a - a)**2
        dV_vals = (dterm1 * term2 + term1 * term2 * dterm2)
        dV[mask_a] = dV_vals
        
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    r_b = dist_safe[mask_b]
    if len(r_b) > 0:
        V_vals_b = B * np.exp(d / (r_b - b))
        dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
        dV[mask_b] = dV_vals_b
        
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return grad

def _is_valid(pos, min_dist=0.5):
    """Check if positions satisfy minimum distance constraint."""
    if len(pos) < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist

def _generate_icosahedron(n_atoms):
    """Generate a structure based on icosahedral shells."""
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)] # Center atom
    
    # Scale radius to fit n_atoms roughly
    # Volume ~ n * (4/3) pi r_min^3. r_min ~ 0.5.
    # Rough estimate: R ~ (n/12)^(1/3) * 1.0
    r_shell = (n_atoms / 12.0) ** (1.0/3.0) * 1.2
    
    for v in verts:
        points.append(r_shell * v)
        
    # Fill remaining atoms randomly inside
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        p = rng.uniform(-r_shell, r_shell, 3)
        # Check distance to existing points
        dists = np.linalg.norm(np.array(points) - p, axis=1)
        if np.min(dists) > 0.5:
            points.append(p)
            
    return np.array(points[:n_atoms])

def _generate_random_cluster(n_atoms, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.0, (n_atoms, 3))
    # Normalize to roughly same volume
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    pos *= (n_atoms ** (1/3)) * 0.8
    return pos

def _local_minimize(pos, time_limit=10.0):
    """
    Use L-BFGS-B to minimize energy.
    Returns (pos, energy, success).
    """
    n = len(pos)
    x0 = pos.flatten()
    
    # Objective function for scipy
    def objective(x):
        p = x.reshape(n, 3)
        if not _is_valid(p):
            return 1e10 # Penalize invalid structures
        return _dzugutov_energy(p)
        
    def jac(x):
        p = x.reshape(n, 3)
        if not _is_valid(p):
            return np.zeros_like(x)
        return _dzugutov_gradient(p).flatten()

    # Check if already valid
    if not _is_valid(pos):
         # Try to fix by random jitter or return as is if we can't fix quickly
         # For simplicity in basin hopping, we might just reject or try a small random move
         # But let's try to relax it first if possible
         pass

    start_time = time.time()
    
    # We use a wrapper to stop if time limit exceeded
    # scipy minimize doesn't have a direct time limit callback that stops cleanly in all versions easily,
    # so we rely on maxiter and a small maxfun/maxiter if time is tight.
    # However, for n=38-60, L-BFGS-B is very fast.
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
        
        if res.success or res.fun < 1e9:
            best_pos = res.x.reshape(n, 3)
            best_energy = res.fun
            if _is_valid(best_pos):
                return best_pos, best_energy, True
            else:
                # If it ended up invalid, try to project back or return original
                return pos, _dzugutov_energy(pos), False
        else:
            return pos, _dzugutov_energy(pos), False
    except Exception as e:
        return pos, _dzugutov_energy(pos), False

def _basin_hopping(pos_init, time_limit, seed):
    """
    Perform Basin Hopping starting from pos_init.
    """
    rng = np.random.default_rng(seed)
    
    # Initial minimization
    pos, energy, _ = _local_minimize(pos_init)
    best_pos = pos.copy()
    best_energy = energy
    
    # Kick magnitude
    kick_size = 0.5
    T = 1.0 # Not strictly temperature, just for acceptance if we wanted Metropolis, 
            # but standard BH accepts if new_min < current_min.
            # Sometimes accepting worse moves helps escape deep traps, but let's stick to strict improvement first.
            
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a random kick
        kick = rng.normal(0, kick_size, pos.shape)
        new_pos = pos + kick
        
        # Ensure validity of the kicked position?
        # If not valid, L-BFGS-B might fail or take long.
        # We can try to relax it.
        
        # Minimize the kicked position
        new_pos_min, new_energy_min, success = _local_minimize(new_pos)
        
        if success and _is_valid(new_pos_min):
            if new_energy_min < best_energy:
                best_energy = new_energy_min
                best_pos = new_pos_min.copy()
                pos = new_pos_min
                # Reduce kick size slightly on success? Or keep constant?
                # Constant is standard.
            elif new_energy_min < energy:
                # Standard BH: accept if new local min is lower than current local min
                pos = new_pos_min
                energy = new_energy_min
            else:
                # Optional: Metropolis acceptance to escape plateaus
                # if rng.random() < np.exp(-(new_energy_min - energy) / T):
                #     pos = new_pos_min
                #     energy = new_energy_min
                pass
        else:
            # If minimization failed or result invalid, try a smaller kick?
            # Or just skip.
            pass
            
        # Adaptive kick size?
        # If we are stuck, increase kick?
        # Simple heuristic: if no improvement in last 5 steps, increase kick.
        # For now, keep it simple.
        
    return best_pos, best_energy

def construct(instance):
    """
    Main entry point.
    """
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # Generate diverse initial guesses
    candidates = []
    
    # 1. Icosahedral seed
    try:
        ico_pos = _generate_icosahedron(n)
        if _is_valid(ico_pos):
            candidates.append(ico_pos)
    except:
        pass
        
    # 2. Random compact seeds
    for i in range(3):
        pos = _generate_random_cluster(n, seed=100+i)
        if _is_valid(pos):
            candidates.append(pos)
            
    # 3. Shell-like random (atoms on a sphere surface)
    rng = np.random.default_rng(200)
    # Fibonacci sphere or random points on sphere
    # Random points on sphere
    v = rng.normal(0, 1, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    radius = (n ** (1/3)) * 0.8
    pos = v * radius
    if _is_valid(pos):
        candidates.append(pos)

    # Sort candidates by initial energy (after a quick relax if needed)
    # To save time, we just evaluate energy directly.
    scored_candidates = []
    for pos in candidates:
        e = _dzugutov_energy(pos)
        scored_candidates.append((e, pos))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    best_overall_energy = np.inf
    best_overall_pos = None
    
    # Allocate time to each candidate
    # We have ~280s. Let's give each candidate a portion.
    # If we have 5 candidates, ~50s each.
    # But Basin Hopping is fast. We can run multiple restarts per candidate or just one long run.
    # Let's run BH on the top 3 candidates, allocating time dynamically.
    
    num_candidates_to_process = min(3, len(scored_candidates))
    time_per_candidate = 250.0 / num_candidates_to_process # Leave 30s buffer
    
    for i in range(num_candidates_to_process):
        if time.time() > deadline - 10.0:
            break
            
        e_init, pos_init = scored_candidates[i]
        
        # Run Basin Hopping
        t_start = time.time()
        pos_bh, e_bh = _basin_hopping(pos_init, time_per_candidate, seed=i)
        t_elapsed = time.time() - t_start
        
        if e_bh < best_overall_energy:
            best_overall_energy = e_bh
            best_overall_pos = pos_bh
            
        # Polish with relax() if time permits
        if time.time() < deadline - 5.0 and best_overall_pos is not None:
            try:
                e_polished, pos_polished = relax(best_overall_pos, 'dzugutov')
                if _is_valid(pos_polished) and e_polished < best_overall_energy:
                    best_overall_energy = e_polished
                    best_overall_pos = pos_polished
            except:
                pass

    if best_overall_pos is None:
        # Fallback: return the best initial candidate
        return scored_candidates[0][1]
        
    return best_overall_pos
# EVOLVE-BLOCK-END
