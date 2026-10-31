"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _objective(params, penalty):
    """
    Objective: maximize sum of radii, penalize overlaps and boundary violations.
    params: (2N,) array [x0,y0,...,x25,y25]
    We maximize sum(r_i) where r_i is determined by the minimum distance to other circles and boundaries.
    But for gradient-based optimization, it's easier to optimize positions AND radii.
    
    Actually, let's use a different approach: optimize positions and radii jointly.
    params: (3N,) array [x0,y0,r0, x1,y1,r1, ...]
    Objective: -sum(r_i) + penalty * (overlap_violations + boundary_violations)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Boundary violations: circle must be inside [0,1]^2
    # x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    bviol = 0.0
    bviol += np.sum(np.maximum(0, -x + r))  # left boundary
    bviol += np.sum(np.maximum(0, x - 1 + r))  # right boundary
    bviol += np.sum(np.maximum(0, -y + r))  # bottom boundary
    bviol += np.sum(np.maximum(0, y - 1 + r))  # top boundary
    
    # Overlap violations: for each pair (i,j), need dist(i,j) >= r_i + r_j
    ov = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                ov += overlap * overlap  # quadratic penalty
    
    return -np.sum(r) + penalty * (bviol + ov)


def _objective_grad(params, penalty):
    """Gradient of the objective for BFGS."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Boundary violations
    # Left: max(0, -x + r) -> d/dx = -1 if -x+r > 0, d/dr = 1 if -x+r > 0
    mask_left = (-x + r) > 0
    grad[0::3][mask_left] += penalty * (-1)
    grad[2::3][mask_left] += penalty * (1)
    
    # Right: max(0, x - 1 + r) -> d/dx = 1 if x-1+r > 0, d/dr = 1 if x-1+r > 0
    mask_right = (x - 1 + r) > 0
    grad[0::3][mask_right] += penalty * (1)
    grad[2::3][mask_right] += penalty * (1)
    
    # Bottom: max(0, -y + r)
    mask_bottom = (-y + r) > 0
    grad[1::3][mask_bottom] += penalty * (-1)
    grad[2::3][mask_bottom] += penalty * (1)
    
    # Top: max(0, y - 1 + r)
    mask_top = (y - 1 + r) > 0
    grad[1::3][mask_top] += penalty * (1)
    grad[2::3][mask_top] += penalty * (1)
    
    # Overlap violations
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                # d/dx_i of (overlap^2) = 2*overlap * d(overlap)/dx_i = 2*overlap * (dx/dist)
                # d/dx_j = 2*overlap * (-dx/dist)
                # d/dr_i = 2*overlap * 1
                # d/dr_j = 2*overlap * 1
                if dist > 1e-12:
                    factor = 2 * penalty * overlap / dist
                    grad[0::3][i] += factor * dx
                    grad[0::3][j] -= factor * dx
                    grad[1::3][i] += factor * dy
                    grad[1::3][j] -= factor * dy
                grad[2::3][i] += 2 * penalty * overlap
                grad[2::3][j] += 2 * penalty * overlap
    
    return grad


def _is_valid(arr):
    """Check if the arrangement is valid: all circles inside [0,1]^2 with disjoint interiors."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Check boundaries
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    if np.any(r < 1e-12):
        return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    
    return True


def _score(arr):
    """Compute the score (sum of radii) if valid, else -inf."""
    if not _is_valid(arr):
        return -np.inf
    return np.sum(arr[:, 2])


def _grid_init(seed=0):
    """Create a grid-based initial configuration."""
    rng = np.random.RandomState(seed)
    side = 6  # 6x6 grid
    positions = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            positions.append((cx, cy))
    # Shuffle and take N
    rng.shuffle(positions)
    positions = positions[:N]
    # Set initial radii
    base_r = 0.5 / side
    params = np.zeros(3*N)
    for k, (cx, cy) in enumerate(positions):
        params[3*k] = cx + rng.uniform(-0.1, 0.1)
        params[3*k+1] = cy + rng.uniform(-0.1, 0.1)
        params[3*k+2] = base_r * 0.9
    return params


def _hex_init(seed=0):
    """Hexagonal packing initial configuration."""
    rng = np.random.RandomState(seed)
    # Hexagonal packing: rows offset
    params = np.zeros(3*N)
    k = 0
    r0 = 0.08
    row = 0
    while k < N:
        n_in_row = N if row % 2 == 0 else N - 1
        for col in range(n_in_row):
            if k >= N:
                break
            x = (col + 0.5) * 2 * r0 + (r0 if row % 2 else 0)
            y = (row + 0.5) * np.sqrt(3) * r0
            # Scale to fit in [0,1]
            x = x / (max(x, 0.01) + 2*r0) if x > 0 else x
            y = y / (max(y, 0.01) + 2*r0) if y > 0 else y
            # Better: just place them and scale
            params[3*k] = 0.1 + 0.8 * (col / max(n_in_row - 1, 1)) + rng.uniform(-0.02, 0.02)
            params[3*k+1] = 0.1 + 0.8 * (row / max(row + 1, 1)) + rng.uniform(-0.02, 0.02)
            params[3*k+2] = r0
            k += 1
        row += 1
    # Ensure all inside
    params[0::3] = np.clip(params[0::3], 0.05, 0.95)
    params[1::3] = np.clip(params[1::3], 0.05, 0.95)
    return params


def _random_init(seed=0):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    params = np.zeros(3*N)
    params[0::3] = rng.uniform(0.1, 0.9, N)
    params[1::3] = rng.uniform(0.1, 0.9, N)
    params[2::3] = rng.uniform(0.05, 0.12, N)
    return params


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Multi-stage optimization with increasing penalty
    penalties = [50, 200, 500, 1000, 2000]
    n_restarts = 12
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization
        if restart % 3 == 0:
            init = _grid_init(seed=restart * 100)
        elif restart % 3 == 1:
            init = _hex_init(seed=restart * 100 + 1)
        else:
            init = _random_init(seed=restart * 100 + 2)
        
        current_params = init.copy()
        
        # Multi-stage penalty optimization
        for penalty in penalties:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _objective,
                    current_params,
                    args=(penalty,),
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    bounds=[(0.01, 0.99)] * N + [(0.01, 0.99)] * N + [(0.001, 0.3)] * N,
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                current_params = result.x
            except Exception:
                pass
        
        # Extract and validate
        x = current_params[0::3]
        y = current_params[1::3]
        r = current_params[2::3]
        
        # Clamp to ensure validity
        r = np.minimum(r, np.minimum(x, 1-x))
        r = np.minimum(r, np.minimum(y, 1-y))
        r = np.maximum(r, 1e-4)
        
        arr = np.column_stack([x, y, r])
        
        # Final polish: try to increase radii while maintaining validity
        # Simple iterative improvement
        for _iter in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try to increase r_i
                max_r = min(x[i], 1-x[i], y[i], 1-y[i])
                for j in range(N):
                    if i != j:
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        max_r = min(max_r, dist - r[j])
                new_r = min(max_r * 0.999, r[i] * 1.05)
                if new_r > r[i] + 1e-8:
                    r[i] = new_r
                    improved = True
            if not improved:
                break
        
        arr = np.column_stack([x, y, r])
        
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_params = arr.copy()
    
    if best_params is not None:
        return best_params
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
