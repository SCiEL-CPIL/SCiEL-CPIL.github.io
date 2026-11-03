"""Circle packing in unit square - tangency-based structural approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import random

def _max_radius_at(x, y, positions, radii, i, eps=1e-12):
    """Compute maximum radius for circle i at position (x,y) given other circles."""
    # Boundary constraint
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    if r_max <= 0:
        return 0.0
    # Disjointness constraints
    for j in range(N):
        if j == i:
            continue
        dx = x - positions[j, 0]
        dy = y - positions[j, 1]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - radii[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(r_max, 0.0)

def _objective_and_grad(x, positions, radii):
    """
    Objective: maximize sum of radii.
    x = [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}] (2N variables)
    For each circle, radius is determined by tightest constraint.
    Returns (-sum_radii, gradient)
    """
    n = N
    total_r = 0.0
    grad = np.zeros(2*n)
    
    for i in range(n):
        xi = x[2*i]
        yi = x[2*i+1]
        
        # Compute max radius at this position
        r_max = min(xi, 1.0 - xi, yi, 1.0 - yi)
        tight_constraint = None  # 'boundary_x', 'boundary_1x', 'boundary_y', 'boundary_1y', or (j, 'circle')
        
        if r_max <= 0:
            total_r += 0.0
            continue
        
        # Check which boundary is tightest
        d_left = xi
        d_right = 1.0 - xi
        d_bottom = yi
        d_top = 1.0 - yi
        
        tight_type = None
        tight_val = r_max
        
        if d_left <= tight_val + 1e-15:
            tight_type = ('left', 0)
            tight_val = d_left
        if d_right < tight_val:
            tight_type = ('right', 0)
            tight_val = d_right
        if d_bottom < tight_val:
            tight_type = ('bottom', 0)
            tight_val = d_bottom
        if d_top < tight_val:
            tight_type = ('top', 0)
            tight_val = d_top
        
        # Check circle constraints
        for j in range(n):
            if j == i:
                continue
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            r_allowed = d - radii[j]
            if r_allowed < tight_val:
                tight_val = r_allowed
                tight_type = ('circle', j)
        
        r_i = max(tight_val, 0.0)
        total_r += r_i
        
        # Gradient contribution
        if r_i <= 0:
            continue
        
        if tight_type is None:
            # Interior, no constraint tight - gradient is 0 for radius w.r.t. position
            # Actually if no constraint is tight, moving doesn't change radius (to first order)
            # But we need to be careful: if multiple constraints are close, gradient is ambiguous
            pass
        elif tight_type[0] == 'left':
            grad[2*i] += 1.0
        elif tight_type[0] == 'right':
            grad[2*i] -= 1.0
        elif tight_type[0] == 'bottom':
            grad[2*i+1] += 1.0
        elif tight_type[0] == 'top':
            grad[2*i+1] -= 1.0
        elif tight_type[0] == 'circle':
            j = tight_type[1]
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-12:
                grad[2*i] += dx / d
                grad[2*i+1] += dy / d
    
    return -total_r, grad

def _compute_radii(positions):
    """Compute optimal radii given fixed positions."""
    radii = np.zeros(N)
    for i in range(N):
        radii[i] = _max_radius_at(positions[i, 0], positions[i, 1], positions, radii, i)
    # Iterate a few times for consistency
    for _ in range(3):
        for i in range(N):
            radii[i] = _max_radius_at(positions[i, 0], positions[i, 1], positions, radii, i)
    return radii

def _is_valid(positions, radii, eps=1e-7):
    """Check if packing is valid."""
    for i in range(N):
        if positions[i, 0] - radii[i] < -eps or positions[i, 0] + radii[i] > 1 + eps:
            return False
        if positions[i, 1] - radii[i] < -eps or positions[i, 1] + radii[i] > 1 + eps:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - eps:
                return False
    return True

def _initial_grid():
    """Grid-based initial positions."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for j in range(side):
        for i in range(side):
            if len(positions) >= N:
                break
            positions.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(positions[:N])

def _initial_random(seed=None):
    """Random initial positions."""
    if seed is not None:
        random.seed(seed)
    positions = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        positions.append([x, y])
    return np.array(positions)

def _optimize_positions(positions0, max_iter=500, tol=1e-10):
    """Optimize positions to maximize sum of radii using L-BFGS-B."""
    x0 = positions0.flatten()
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 2)
        radii = _compute_radii(pos)
        # Recompute objective with these radii
        total_r = 0.0
        grad = np.zeros(2*N)
        for i in range(N):
            xi = pos[i, 0]
            yi = pos[i, 1]
            r_max = min(xi, 1.0 - xi, yi, 1.0 - yi)
            tight_type = None
            tight_val = r_max
            
            if d_left := xi <= tight_val + 1e-15:
                tight_type = ('left',)
                tight_val = xi
            if d_right := 1.0 - xi < tight_val:
                tight_type = ('right',)
                tight_val = 1.0 - xi
            if d_bottom := yi < tight_val:
                tight_type = ('bottom',)
                tight_val = yi
            if d_top := 1.0 - yi < tight_val:
                tight_type = ('top',)
                tight_val = 1.0 - yi
            
            for j in range(N):
                if j == i:
                    continue
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - radii[j]
                if r_allowed < tight_val:
                    tight_val = r_allowed
                    tight_type = ('circle', j)
            
            r_i = max(tight_val, 0.0)
            total_r += r_i
            
            if r_i <= 0 or tight_type is None:
                continue
            
            if tight_type[0] == 'left':
                grad[2*i] += 1.0
            elif tight_type[0] == 'right':
                grad[2*i] -= 1.0
            elif tight_type[0] == 'bottom':
                grad[2*i+1] += 1.0
            elif tight_type[0] == 'top':
                grad[2*i+1] -= 1.0
            elif tight_type[0] == 'circle':
                j = tight_type[1]
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-12:
                    grad[2*i] += dx / d
                    grad[2*i+1] += dy / d
        
        return -total_r, grad
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-12})
    
    return res.x.reshape(-1, 2), res.fun, res

def _local_search_positions(positions, max_iter=100):
    """Simple local search: try small perturbations."""
    positions = positions.copy()
    best_sum = sum(_compute_radii(positions))
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for ss in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for dx in [-ss, 0, ss]:
                    for dy in [-ss, 0, ss]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = positions.copy()
                        new_pos[i, 0] = np.clip(positions[i, 0] + dx, 0.01, 0.99)
                        new_pos[i, 1] = np.clip(positions[i, 1] + dy, 0.01, 0.99)
                        radii = _compute_radii(new_pos)
                        s = sum(radii)
                        if s > best_sum + 1e-10:
                            positions = new_pos
                            best_sum = s
                            improved = True
    return positions, best_sum

def _high_precision_polish(positions):
    """Final high-precision optimization."""
    x0 = positions.flatten()
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 2)
        radii = _compute_radii(pos)
        total_r = 0.0
        grad = np.zeros(2*N)
        for i in range(N):
            xi = pos[i, 0]
            yi = pos[i, 1]
            r_max = min(xi, 1.0 - xi, yi, 1.0 - yi)
            tight_type = None
            tight_val = r_max
            
            if xi <= tight_val + 1e-15:
                tight_type = ('left',)
                tight_val = xi
            if 1.0 - xi < tight_val:
                tight_type = ('right',)
                tight_val = 1.0 - xi
            if yi < tight_val:
                tight_type = ('bottom',)
                tight_val = yi
            if 1.0 - yi < tight_val:
                tight_type = ('top',)
                tight_val = 1.0 - yi
            
            for j in range(N):
                if j == i:
                    continue
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - radii[j]
                if r_allowed < tight_val:
                    tight_val = r_allowed
                    tight_type = ('circle', j)
            
            r_i = max(tight_val, 0.0)
            total_r += r_i
            
            if r_i <= 0 or tight_type is None:
                continue
            
            if tight_type[0] == 'left':
                grad[2*i] += 1.0
            elif tight_type[0] == 'right':
                grad[2*i] -= 1.0
            elif tight_type[0] == 'bottom':
                grad[2*i+1] += 1.0
            elif tight_type[0] == 'top':
                grad[2*i+1] -= 1.0
            elif tight_type[0] == 'circle':
                j = tight_type[1]
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-12:
                    grad[2*i] += dx / d
                    grad[2*i+1] += dy / d
        
        return -total_r, grad
    
    bounds = [(0.0, 1.0)] * (2 * N)
    res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                   bounds=bounds, options={'maxiter': 200, 'gtol': 1e-14, 'ftol': 1e-15})
    
    return res.x.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    best_positions = None
    best_sum = 0.0
    
    # Multiple initializations
    initializations = [_initial_grid()]
    for seed in range(5):
        initializations.append(_initial_random(seed))
    
    for init_pos in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            # Coarse optimization
            pos, fun_val, res = _optimize_positions(init_pos, max_iter=200, tol=1e-8)
            radii = _compute_radii(pos)
            s = sum(radii)
            
            if s > best_sum:
                best_sum = s
                best_positions = pos.copy()
            
            # Local search
            if time.time() > deadline - 15:
                break
            pos_ls, s_ls = _local_search_positions(pos, max_iter=50)
            if s_ls > best_sum:
                best_sum = s_ls
                best_positions = pos_ls.copy()
            
            # High precision polish
            if time.time() > deadline - 10:
                break
            pos_hp = _high_precision_polish(best_positions)
            radii_hp = _compute_radii(pos_hp)
            s_hp = sum(radii_hp)
            if s_hp > best_sum:
                best_sum = s_hp
                best_positions = pos_hp.copy()
                
        except Exception:
            continue
    
    # Final verification and return
    if best_positions is None:
        best_positions = _initial_grid()
    
    radii = _compute_radii(best_positions)
    
    # Verify validity
    if not _is_valid(best_positions, radii):
        # Fallback to grid
        best_positions = _initial_grid()
        radii = _compute_radii(best_positions)
    
    result = np.column_stack([best_positions, radii])
    return result
# EVOLVE-BLOCK-END
