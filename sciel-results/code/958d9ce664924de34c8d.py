"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _pairwise_min_gap(dist_matrix, radii):
    """For each pair (i,j), the gap is dist - r_i - r_j. We want all gaps >= 0."""
    # dist_matrix is (n,n), radii is (n,)
    # gap[i,j] = dist[i,j] - radii[i] - radii[j]
    return dist_matrix - radii[:, None] - radii[None, :]

def _smooth_max(x, beta=100.0):
    """Smooth maximum via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    # Use a stable log-sum-exp
    shifted = x - x_max
    s = np.sum(np.exp(beta * shifted))
    if s < 1e-300:
        return x_max
    return x_max + np.log(s) / beta

def _objective(params, beta=50.0):
    """
    params: (2n + n) = 3n parameters: x (n), y (n), r (n)
    Objective: -sum(r) + penalty * smooth_max(0, -min_gap) + penalty_wall * smooth_max(0, wall_violations)
    """
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = np.maximum(params[2*n:3*n], 1e-6)  # ensure positive radii
    
    # Compute pairwise distances
    coords = np.column_stack([x, y])
    dist = cdist(coords, coords)
    
    # Gaps: dist[i,j] - r[i] - r[j]
    gaps = dist - r[:, None] - r[None, None]
    
    # Wall violations: 
    # x_i - r_i < 0 => r_i - x_i > 0
    # 1 - x_i - r_i < 0 => x_i + r_i - 1 > 0
    # y_i - r_i < 0 => r_i - y_i > 0
    # 1 - y_i - r_i < 0 => y_i + r_i - 1 > 0
    wall_violations = np.concatenate([
        r - x,        # left wall
        x + r - 1.0,  # right wall
        r - y,        # bottom wall
        y + r - 1.0   # top wall
    ])
    
    # We want all gaps >= 0 and all wall_violations <= 0
    # Penalty for gaps < 0: smooth_max(0, -gaps)
    # Penalty for wall_violations > 0: smooth_max(0, wall_violations)
    
    # Only consider upper triangle for gaps to avoid double counting
    iu = np.triu_indices(n, k=1)
    gap_violations = -gaps[iu]  # positive if overlapping
    
    # Combine penalties
    overlap_penalty = _smooth_max(np.concatenate([gap_violations, np.zeros(1)]), beta=beta)
    wall_penalty = _smooth_max(np.concatenate([wall_violations, np.zeros(1)]), beta=beta)
    
    # Objective: maximize sum(r) -> minimize -sum(r) + penalties
    obj = -np.sum(r) + 10.0 * overlap_penalty + 10.0 * wall_penalty
    
    return obj

def _grad(params, beta=50.0):
    """
    Numerical gradient approximation (or analytical if feasible).
    For simplicity, use finite differences with central differences.
    """
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (_objective(p_plus, beta) - _objective(p_minus, beta)) / (2 * eps)
    return grad

def _initial_grid():
    """Generate an initial grid configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append((x, y, r))
    return np.array(cells[:N])

def _random_initialization(seed=42):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    n = N
    # Random positions in [0.1, 0.9] to start away from walls
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    # Small initial radii
    r = rng.uniform(0.02, 0.05, n)
    return np.concatenate([x, y, r])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_params = None
    best_score = -np.inf
    
    # Start with grid initialization
    grid = _initial_grid()
    init_params = np.concatenate([grid[:, 0], grid[:, 1], grid[:, 2]])
    
    # Try multiple random restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        if seed == 42:
            params = init_params.copy()
        else:
            params = _random_initialization(seed)
        
        # Run optimization with L-BFGS-B
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                jac=_grad,
                bounds=[(1e-6, 1.0)] * N + [(1e-6, 1.0)] * N + [(1e-6, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
                
        except Exception:
            continue
    
    # If optimization didn't improve, fall back to grid
    if best_params is None:
        best_params = init_params
    
    # Final polish: try to increase radii slightly while maintaining feasibility
    # Use a simple iterative adjustment
    x = best_params[:N].copy()
    y = best_params[N:2*N].copy()
    r = best_params[2*N:3*N].copy()
    
    # Try a few iterations of radius expansion
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to increase r[i] slightly
            old_r = r[i]
            # Check if we can increase
            new_r = old_r * 1.01 + 1e-4
            if new_r > 0.5:
                new_r = 0.5
            if new_r <= old_r:
                continue
            
            # Check constraints
            feasible = True
            # Wall constraints
            if x[i] - new_r < -1e-9 or 1.0 - x[i] - new_r < -1e-9:
                feasible = False
            if y[i] - new_r < -1e-9 or 1.0 - y[i] - new_r < -1e-9:
                feasible = False
            
            if feasible:
                # Check overlap with other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_r + r[j] - 1e-9:
                        feasible = False
                        break
            
            if feasible:
                r[i] = new_r
                improved = True
        
        if not improved:
            break
    
    # Construct final result
    result = np.column_stack([x, y, r])
    
    # Final validity check
    valid = True
    for i in range(N):
        xi, yi, ri = result[i]
        if xi < -1e-9 or xi > 1.0 + 1e-9 or yi < -1e-9 or yi > 1.0 + 1e-9 or ri < 1e-9:
            valid = False
            break
        if xi - ri < -1e-9 or 1.0 - xi - ri < -1e-9 or yi - ri < -1e-9 or 1.0 - yi - ri < -1e-9:
            valid = False
            break
    
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < result[i, 2] + result[j, 2] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fall back to the known-valid grid
        grid = _initial_grid()
        result = grid
    
    return result
# EVOLVE-BLOCK-END
