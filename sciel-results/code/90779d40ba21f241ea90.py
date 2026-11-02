"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26
PENALTY_SCALE = 1000.0  # Increased penalty to enforce constraints strictly during local opt

def _objective_and_grad(params):
    """Objective and gradient for optimization.
    
    Returns (objective_value, gradient).
    Objective = penalty - sum(radii), so minimizing maximizes radii.
    """
    n = N
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:].copy()
    
    # Objective value
    obj = 0.0
    
    # Boundary penalties
    for i in range(n):
        # x-boundary: violation = r[i] - x[i] if x[i] < r[i]
        viol_x_left = max(0, r[i] - x[i])
        viol_x_right = max(0, r[i] - (1 - x[i]))
        viol_y_bottom = max(0, r[i] - y[i])
        viol_y_top = max(0, r[i] - (1 - y[i]))
        
        obj += PENALTY_SCALE * (viol_x_left**2 + viol_x_right**2 + viol_y_bottom**2 + viol_y_top**2)
    
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = r[i] + r[j]
            overlap = min_d - d
            if overlap > 0:
                obj += PENALTY_SCALE * overlap * overlap
    
    # Negative sum of radii (we want to maximize sum(r))
    obj -= np.sum(r)
    
    # Gradient
    grad = np.zeros_like(params)
    
    # Gradient of boundary penalties w.r.t. x, y, r
    for i in range(n):
        # d/dx of PENALTY_SCALE * (max(0, r-x)^2 + max(0, r-(1-x))^2)
        if x[i] < r[i]:
            grad[i] += PENALTY_SCALE * 2 * (r[i] - x[i]) * (-1)  # d/dx of (r-x)^2
        if x[i] > 1 - r[i]:
            grad[i] += PENALTY_SCALE * 2 * (r[i] - (1 - x[i])) * (1)  # d/dx of (r-1+x)^2
        
        # d/dy
        idx_y = n + i
        if y[i] < r[i]:
            grad[idx_y] += PENALTY_SCALE * 2 * (r[i] - y[i]) * (-1)
        if y[i] > 1 - r[i]:
            grad[idx_y] += PENALTY_SCALE * 2 * (r[i] - (1 - y[i])) * (1)
        
        # d/dr of boundary penalties
        idx_r = 2*n + i
        if x[i] < r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - x[i]) * (1)
        if x[i] > 1 - r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - (1 - x[i])) * (1)
        if y[i] < r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - y[i]) * (1)
        if y[i] > 1 - r[i]:
            grad[idx_r] += PENALTY_SCALE * 2 * (r[i] - (1 - y[i])) * (1)
        
        # d/dr of -sum(r)
        grad[idx_r] += -1
    
    # Gradient of pairwise penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = r[i] + r[j]
            overlap = min_d - d
            
            if overlap > 0:
                factor = PENALTY_SCALE * 2 * overlap
                grad[i] += factor * (-dx / d)       # d/dx_i
                grad[n + i] += factor * (-dy / d)   # d/dy_i
                grad[2*n + i] += factor * (1)       # d/dr_i
                grad[2*n + j] += factor * (1)       # d/dr_j
                
                # For circle j:
                grad[j] += factor * (dx / d)        # d/dx_j
                grad[n + j] += factor * (dy / d)    # d/dy_j
    
    return obj, grad

def _compute_score(params):
    """Compute the actual score (sum of radii) if feasible, else -inf."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Check boundaries
    for i in range(n):
        if x[i] < -1e-9 or x[i] > 1 + 1e-9 or y[i] < -1e-9 or y[i] > 1 + 1e-9:
            return -np.inf
        if r[i] < 0:
            return -np.inf
        if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
            return -np.inf
        if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
            return -np.inf
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - 1e-9:
                return -np.inf
    
    return np.sum(r)

def _enforce_validity(params):
    """Clip parameters to ensure validity."""
    n = N
    x = params[:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:].copy()
    
    # Ensure radii are positive and reasonable
    r = np.clip(r, 1e-6, 0.5)
    
    # Ensure circles are inside the square
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.concatenate([x, y, r])

def _local_optimize(params, deadline):
    """Run L-BFGS-B optimization."""
    bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
    
    def obj_wrapper(p):
        val, _ = _objective_and_grad(p)
        return val
    
    def grad_wrapper(p):
        _, g = _objective_and_grad(p)
        return g
    
    try:
        result = minimize(
            obj_wrapper,
            params,
            jac=grad_wrapper,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        return result.x
    except Exception:
        return params

def _random_initialization():
    """Generate a random initialization."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.03, 0.15, N)
    return np.concatenate([x, y, r])

def _hexagonal_initialization():
    """Generate a hexagonal initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # 5 rows, 5 cols = 25, plus 1 in center
    # Row 0: 5 circles
    # Row 1: 5 circles (offset)
    # ...
    
    # Simpler: 4x7 grid = 28, take 26
    # Or 5x5 grid = 25, plus 1
    
    # Let's do 5x5 grid + 1
    side = 5
    count = 0
    for row in range(side):
        for col in range(side):
            if count >= N:
                break
            x[count] = (col + 0.5) / side
            y[count] = (row + 0.5) / side
            r[count] = 0.5 / side - 0.01
            count += 1
        if count >= N:
            break
    
    if count < N:
        # Add remaining in the center
        for i in range(count, N):
            x[i] = 0.5
            y[i] = 0.5
            r[i] = 0.05
    
    return np.concatenate([x, y, r])

def _grid_initialization(n_cols, n_rows):
    """Initialize circles in a grid pattern."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x[count] = (col + 0.5) / n_cols
            y[count] = (row + 0.5) / n_rows
            r[count] = 0.5 / max(n_cols, n_rows) - 1e-4
            count += 1
    
    # Fill remaining with small circles in gaps
    if count < N:
        for i in range(count, N):
            angle = 2 * np.pi * (i - count) / max(1, N - count)
            x[i] = 0.5 + 0.15 * np.cos(angle)
            y[i] = 0.5 + 0.15 * np.sin(angle)
            r[i] = 0.04
    
    return np.concatenate([x, y, r])

def construct():
    """Construct an arrangement of 26 circles in the unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_params = None
    best_score = -np.inf
    
    # Strategy: Multistart Local Search
    # 1. Generate diverse initializations
    # 2. Run local optimization on each
    # 3. Keep the best
    
    initializations = [
        _random_initialization(),
        _hexagonal_initialization(),
        _grid_initialization(5, 6),
        _grid_initialization(6, 5),
        _grid_initialization(4, 7),
        _grid_initialization(7, 4),
    ]
    
    # Add more random restarts
    for _ in range(10):
        initializations.append(_random_initialization())
    
    for init_params in initializations:
        if time.time() > deadline - 10:
            break
        
        # Local optimization
        params1 = _local_optimize(init_params, deadline)
        score = _compute_score(params1)
        
        if score > best_score:
            best_score = score
            best_params = params1.copy()
            
    # If we have a good solution, try to improve it further with perturbations
    if best_params is not None and best_score > 0:
        # Simulated Annealing-like perturbations
        current_params = best_params.copy()
        current_score = best_score
        temperature = 0.05
        cooling_rate = 0.95
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            noise = np.random.normal(0, temperature, 3 * N)
            perturbed = current_params + noise
            perturbed = _enforce_validity(perturbed)
            
            # Local optimization
            new_params = _local_optimize(perturbed, deadline)
            new_score = _compute_score(new_params)
            
            # Acceptance criterion
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                # Accept with probability exp(-delta / T)
                delta = current_score - new_score
                if delta < 0:
                    # New is better, already handled
                    pass
                else:
                    prob = np.exp(-delta / (temperature + 1e-6))
                    if np.random.rand() < prob:
                        current_params = new_params
                        current_score = new_score
            
            temperature *= cooling_rate
    
    # Final polish
    if best_params is not None and time.time() < deadline - 2:
        # Run a tighter optimization
        bounds = [(0, 1)] * (2 * N) + [(0.01, 0.5)] * N
        
        def obj_wrapper(p):
            val, _ = _objective_and_grad(p)
            return val
        
        def grad_wrapper(p):
            _, g = _objective_and_grad(p)
            return g
        
        try:
            result = minimize(
                obj_wrapper,
                best_params,
                jac=grad_wrapper,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            final_params = result.x
            final_score = _compute_score(final_params)
            if final_score > best_score:
                best_params = final_params
                best_score = final_score
        except Exception:
            pass
    
    # Return the best valid solution
    if best_params is not None and best_score > 0:
        best_params = _enforce_validity(best_params)
        n = N
        x = best_params[:n]
        y = best_params[n:2*n]
        r = best_params[2*n:]
        return np.column_stack([x, y, r])
    
    # Fallback: return a valid grid arrangement
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    count = 0
    for row in range(side):
        for col in range(side):
            if count >= N:
                break
            x_val = (col + 0.5) / side
            y_val = (row + 0.5) / side
            r_val = 0.5 / side - 1e-3
            cells.append([x_val, y_val, r_val])
            count += 1
    
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
