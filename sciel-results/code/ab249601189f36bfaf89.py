"""
Morse cluster global minima search using Basin Hopping with crystalline seeds.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """Compute Morse energy exactly as specified."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))

def _generate_fcc_seed(n, spacing=1.0):
    """Generate a truncated FCC lattice seed."""
    points = []
    # FCC lattice vectors
    a = spacing
    # Generate points in a cubic grid of FCC sites
    # FCC basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    max_idx = int(math.ceil(math.sqrt(n))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for dx, dy, dz in [(0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + dx, j*a + dy, k*a + dz])
                    if len(points) < n:
                        points.append(p)
    points = np.array(points[:n])
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points

def _generate_bcc_seed(n, spacing=1.0):
    """Generate a truncated BCC lattice seed."""
    points = []
    a = spacing
    max_idx = int(math.ceil(math.sqrt(n))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC basis: (0,0,0), (a/2, a/2, a/2)
                for dx, dy, dz in [(0,0,0), (a/2, a/2, a/2)]:
                    p = np.array([i*a + dx, j*a + dy, k*a + dz])
                    if len(points) < n:
                        points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _local_minimize(pos, rho, max_iter=200):
    """Local minimization using L-BFGS-B."""
    def func_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = morse_energy_gradient(p, rho)
        return e, g.flatten()
    
    try:
        from scipy.optimize import minimize
        x0 = pos.flatten()
        res = minimize(func_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.status in [0, 1, 2]:
            return res.fun, res.x.reshape(-1, 3)
        else:
            # If minimization fails, return the last evaluated point or original
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos

def _basin_hopping_step(pos, rho, kick_size, rng):
    """Perform one basin hopping step: kick, minimize, accept/reject."""
    # Kick
    kick = rng.normal(0, kick_size, pos.shape)
    new_pos = pos + kick
    
    # Minimize
    new_e, new_pos_min = _local_minimize(new_pos, rho)
    
    return new_e, new_pos_min

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time management
    t_start = time.time()
    deadline = t_start + 325.0  # 5s safety margin
    
    # Generate diverse seeds
    seeds = []
    # FCC is usually better for Morse
    seeds.append(_generate_fcc_seed(n))
    seeds.append(_generate_bcc_seed(n))
    
    # Add a couple of perturbed FCC seeds
    rng = np.random.default_rng(42)
    for _ in range(2):
        seed = _generate_fcc_seed(n)
        seed += rng.normal(0, 0.1, seed.shape)
        seeds.append(seed)
        
    # Initial minimization of seeds
    best_e = np.inf
    best_pos = None
    
    for seed in seeds:
        if time.time() > deadline:
            break
        e, pos = _local_minimize(seed, rho)
        if e < best_e:
            best_e = e
            best_pos = pos
    
    if best_pos is None:
        # Fallback if something went wrong
        return _generate_fcc_seed(n)
        
    # Basin Hopping Loop
    # Kick size heuristic: start moderate, decrease slightly
    kick_size = 0.2
    # Temperature schedule
    # T should be comparable to energy differences between minima
    # For Morse, energy scale is ~1.0 per pair, total ~N.
    # Let's start T around 1.0 and cool down.
    T = 1.0
    cooling_rate = 0.99
    
    # Track current state
    curr_e = best_e
    curr_pos = best_pos
    
    # Determine number of steps based on time
    # Each step involves a minimization. For N=80, this might take ~5-10ms.
    # Let's just loop until time runs out.
    
    while time.time() < deadline:
        # Generate kick
        kick = rng.normal(0, kick_size, curr_pos.shape)
        new_pos = curr_pos + kick
        
        # Minimize
        new_e, new_pos_min = _local_minimize(new_pos, rho)
        
        # Metropolis acceptance
        delta_e = new_e - curr_e
        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            curr_e = new_e
            curr_pos = new_pos_min
            
            if curr_e < best_e:
                best_e = curr_e
                best_pos = curr_pos
                
        # Cooling
        T *= cooling_rate
        
        # Adapt kick size: if we've found a new best, maybe reduce kick? 
        # Or keep it constant? Constant is standard. 
        # Let's slowly reduce kick size to refine.
        kick_size *= 0.999
        
    return best_pos
# EVOLVE-BLOCK-END
