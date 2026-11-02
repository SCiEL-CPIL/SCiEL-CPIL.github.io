"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 26
# Safety margin: 150s limit, aim for 140s usage
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

# EVOLVE-BLOCK-START

def _time_remaining():
    return _DEADLINE - time.time()

def _softplus(z, eps=1e-4):
    """Smooth max(0, z)."""
    return np.logaddexp(0, z / eps) * eps

def _energy(params, penalty_weight, eps=1e-4):
    """
    Calculate energy: -sum(r) + penalty_weight * penalty.
    params: flat array [x0, y0, r0, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # 1. Negative radius penalty
    penalty = np.sum(_softplus(-r, eps)**2)
    
    # 2. Boundary penalties
    # x - r < 0  =>  r - x > 0
    penalty += np.sum(_softplus(r - x, eps)**2)
    # x + r > 1  =>  x + r - 1 > 0
    penalty += np.sum(_softplus(x + r - 1.0, eps)**2)
    # y - r < 0
    penalty += np.sum(_softplus(r - y, eps)**2)
    # y + r > 1
    penalty += np.sum(_softplus(y + r - 1.0, eps)**2)
    
    # 3. Overlap penalties (Vectorized full pairwise)
    # For N=26, this is 325 pairs. Very fast.
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _is_valid(xyz, tol=1e-7):
    """Check if configuration is strictly valid."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if np.any(r < 0): return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol): return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol): return False
    
    # Pairwise check
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(dist < r[idx_i] + r[idx_j] - tol):
        return False
    return True

def _fix_validity(xyz, tol=1e-7):
    """
    Attempt to make a configuration valid by slightly shrinking radii
    and projecting centers into bounds.
    """
    xyz = xyz.copy()
    # Project centers into bounds given current radii
    for i in range(N):
        r = max(0, xyz[i, 2])
        xyz[i, 0] = np.clip(xyz[i, 0], r, 1-r)
        xyz[i, 1] = np.clip(xyz[i, 1], r, 1-r)
        xyz[i, 2] = r
        
    # Iteratively shrink radii if overlaps persist
    for _ in range(50):
        if _is_valid(xyz, tol=tol):
            break
        # Find max overlap
        x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist = np.sqrt(dx*dx + dy*dy)
        overlaps = r[idx_i] + r[idx_j] - dist
        max_overlap = np.max(overlaps)
        if max_overlap <= tol:
            # Only boundary issues?
            if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
                 # Shrink all radii slightly to allow center movement?
                 # Or just accept if boundary violation is small?
                 # Let's shrink radii a bit
                 xyz[:, 2] *= 0.99
            else:
                 break
        else:
            # Shrink radii involved in max overlap or all
            # Simple strategy: shrink all radii by a factor
            factor = 1.0 - max_overlap * 0.5 - 0.01
            xyz[:, 2] *= max(0.9, factor)
            
    return xyz

def _initial_grid(seed=0):
    rng = np.random.default_rng(seed)
    side = 6 # 6x6 = 36 slots, take 26
    positions = []
    # Start with small radii to ensure validity
    r_init = 0.05
    for i in range(6):
        for j in range(6):
            if len(positions) >= N: break
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            # Add small noise
            x += rng.uniform(-0.01, 0.01)
            y += rng.uniform(-0.01, 0.01)
            positions.append([x, y, r_init])
    return np.array(positions[:N])

def _initial_random(seed=0):
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(N):
        r = rng.uniform(0.05, 0.15)
        x = rng.uniform(r, 1-r)
        y = rng.uniform(r, 1-r)
        positions.append([x, y, r])
    return np.array(positions)

def _initial_corner(seed=0):
    """Place large circles in corners, small in center."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.12])
    # Fill rest
    for _ in range(N - 4):
        r = rng.uniform(0.05, 0.1)
        x = rng.uniform(r, 1-r)
        y = rng.uniform(r, 1-r)
        positions.append([x, y, r])
    return np.array(positions)

def _simulated_annealing(xyz0, time_budget):
    """
    Run SA on the configuration.
    """
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_energy = _energy(params, 100.0, 1e-4) # Initial energy with moderate penalty
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-6
    cooling_rate = 0.999 # Adjusted for number of steps
    
    # Dynamic penalty weight
    # Start with low penalty to explore, increase to enforce constraints
    penalty_weight = 10.0
    
    n_steps = 0
    start_time = time.time()
    
    # Estimate steps possible
    # _energy is ~10-50 us? Let's assume 20us.
    # 10s / 20us = 500k steps.
    
    while _time_remaining() > 5.0 and (time.time() - start_time) < time_budget:
        # Adaptive penalty
        # If penalty is high, increase weight
        current_penalty = best_energy + np.sum(params[2::3])
        if current_penalty > 1e-3:
            penalty_weight *= 1.01
        else:
            penalty_weight = max(1.0, penalty_weight * 0.99)
            
        # Pick a circle
        i = np.random.randint(N)
        
        # Step size: decrease as T decreases? Or keep constant?
        # Constant step size is often better for packing.
        step_pos = 0.05
        step_rad = 0.02
        
        dx = np.random.normal(0, step_pos)
        dy = np.random.normal(0, step_pos)
        dr = np.random.normal(0, step_rad)
        
        new_params = params.copy()
        new_params[i*3] += dx
        new_params[i*3+1] += dy
        new_params[i*3+2] += dr
        
        # Enforce bounds roughly to keep search in box
        # x in [0,1], y in [0,1], r in [0, 0.5]
        if new_params[i*3] < 0: new_params[i*3] = 0
        if new_params[i*3] > 1: new_params[i*3] = 1
        if new_params[i*3+1] < 0: new_params[i*3+1] = 0
        if new_params[i*3+1] > 1: new_params[i*3+1] = 1
        if new_params[i*3+2] < 0: new_params[i*3+2] = 0
        if new_params[i*3+2] > 0.5: new_params[i*3+2] = 0.5
        
        new_energy = _energy(new_params, penalty_weight, 1e-4)
        delta_e = new_energy - _energy(params, penalty_weight, 1e-4)
        
        # Acceptance
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_start):
            params = new_params
            # Update best
            # Use a fixed high penalty for best tracking to ensure validity preference
            # Or just track sum(r) if penalty is low?
            # Let's track the actual objective: -sum(r) + high_penalty * penalty
            # If penalty is ~0, this is -sum(r).
            check_energy = _energy(params, 1000.0, 1e-4)
            if check_energy < best_energy:
                best_energy = check_energy
                best_params = params.copy()
                
        # Cool down
        T_start *= cooling_rate
        n_steps += 1
        
        # Reset temperature if stuck? (Restart SA)
        # Simple: if no improvement for many steps, reheat?
        # For now, linear cooling is safer.
        
    return best_params.reshape(N, 3)

def _lbfgs_polish(xyz0, time_budget=5.0):
    """
    Polish a configuration using L-BFGS-B.
    """
    params = xyz0.flatten().copy()
    
    # Use a high penalty weight to ensure we end up in a valid region
    # The smooth penalty helps L-BFGS navigate.
    penalty_weight = 1000.0
    eps = 1e-5
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # We want to minimize energy.
    # Objective: -sum(r) + penalty
    
    def obj(p):
        return _energy(p, penalty_weight, eps)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < obj(params):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return xyz0

def construct():
    best_xyz = None
    best_score = 0.0
    
    # Initial guesses
    inits = [
        _initial_grid(0),
        _initial_random(1),
        _initial_corner(2),
        _initial_grid(3),
        _initial_random(4),
    ]
    
    # Time allocation
    # Total 140s.
    # Leave 10s for final polish and validation.
    # Remaining 130s for SA trials.
    # 5 trials -> 26s each.
    
    for i, xyz0 in enumerate(inits):
        if _time_remaining() < 15.0:
            break
            
        # SA Phase
        sa_time = 20.0 # 20s per SA trial
        if _time_remaining() < sa_time + 10.0:
            sa_time = _time_remaining() - 10.0
            
        if sa_time > 2.0:
            xyz_sa = _simulated_annealing(xyz0, sa_time)
            
            # L-BFGS Polish Phase
            if _time_remaining() > 5.0:
                xyz_polished = _lbfgs_polish(xyz_sa, time_budget=5.0)
            else:
                xyz_polished = xyz_sa
            
            # Ensure validity
            xyz_valid = _fix_validity(xyz_polished)
            score = np.sum(xyz_valid[:, 2])
            
            if score > best_score:
                best_score = score
                best_xyz = xyz_valid
                
    # If we have time left, do a final random restart + SA
    if _time_remaining() > 25.0 and best_xyz is not None:
        # Use best solution as seed for further improvement?
        # Or try a new random start.
        xyz_new = _initial_random(100)
        # Perturb best solution
        rng = np.random.default_rng(123)
        noise = rng.normal(0, 0.05, best_xyz.shape)
        noise[:, 2] *= 0.1
        xyz_perturbed = best_xyz + noise
        xyz_perturbed = _fix_validity(xyz_perturbed)
        
        sa_time = _time_remaining() - 10.0
        if sa_time > 5.0:
            xyz_sa = _simulated_annealing(xyz_perturbed, sa_time)
            xyz_polished = _lbfgs_polish(xyz_sa, time_budget=5.0)
            xyz_valid = _fix_validity(xyz_polished)
            score = np.sum(xyz_valid[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_valid

    # Final Fallback
    if best_xyz is None or not _is_valid(best_xyz):
        # Return a known valid configuration
        # Simple grid
        best_xyz = _initial_grid(0)
        # Shrink to ensure validity
        for _ in range(100):
            if _is_valid(best_xyz):
                break
            best_xyz[:, 2] *= 0.95
            best_xyz = _fix_validity(best_xyz)
            
    return best_xyz

# EVOLVE-BLOCK-END
