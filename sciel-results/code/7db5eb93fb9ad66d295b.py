"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Check inside square
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-4)
    
    # Smooth penalty using soft-plus (log-sum-exp trick) for boundary violations
    # This gives a smooth approximation to max(0, violation)^2
    penalty = 0.0
    for i in range(n):
        # Left boundary: violation = radii[i] - pos[i, 0]
        v = radii[i] - pos[i, 0]
        if v > 0:
            penalty += v * v
        # Right boundary: violation = pos[i, 0] - (1.0 - radii[i])
        v = pos[i, 0] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
        # Bottom boundary
        v = radii[i] - pos[i, 1]
        if v > 0:
            penalty += v * v
        # Top boundary
        v = pos[i, 1] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
    
    # Penalty for overlapping - use smooth penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            # Use dist^2 to avoid sqrt in the hot loop
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                v = min_dist - dist
                penalty += v * v
    
    return -np.sum(radii) + 5000.0 * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid: 25 circles
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    # 26th circle: place in a corner
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    # Row 1: 4 circles (offset)
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    # Row 3: 4 circles (offset)
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    # 26th circle
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        # Check and fix boundaries
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            # Right
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Bottom
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Top
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        # Check and fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision local search to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((1e-5, 0.5))  # radius
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple polish passes with different methods and tolerances
    polish_configs = [
        ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}),
        ('Powell', {'maxiter': 500, 'ftol': 1e-12}),
        ('L-BFGS-B', {'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12}),
    ]
    
    for method, options in polish_configs:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, method=method, bounds=bounds,
                              options=options)
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                # Post-process to ensure validity
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    # Use improved solution as new starting point
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
        
    return best_pos, best_radii, best_score

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.01, 0.30))  # radius
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try multiple optimization methods with progressive refinement
    method_configs = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-10}),
        ('Powell', {'maxiter': 300, 'ftol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 800, 'ftol': 1e-12}),
    ]
    
    for method, options in method_configs:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(_objective, x0, method=method, bounds=bounds,
                              options=options)
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                # Post-process to ensure validity
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    # Use improved solution as new starting point
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
            
    # Final polish step if time permits
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Try different initial configurations
    init_configs = []
    
    # Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii, 'grid'))
    
    # Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii, 'hex'))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 101]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii, f'rand_{seed}'))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii, 'grid_pert'))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii, 'hex_pert'))
    
    # Run optimization from each initial config
    for pos, radii, name in init_configs:
        if time.time() > deadline - 20:
            break
        
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None or best_score < 0:
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx, 0] = (i + 0.5) / side
                    pos[idx, 1] = (j + 0.5) / side
                    radii[idx] = 0.09
        # 26th circle
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # One final high-precision polish if time permits
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        # Ensure final validity
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
