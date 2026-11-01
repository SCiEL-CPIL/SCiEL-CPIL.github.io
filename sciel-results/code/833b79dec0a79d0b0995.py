"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32
DEADLINE = 110.0  # Safety margin before 120s limit


def _compute_penalty(x, y, r):
    """Compute smooth penalty for constraint violations."""
    # Boundary violations: circles must be inside [0,1]^2
    # x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    boundary_pen = 0.0
    for i in range(N):
        boundary_pen += max(0, r[i] - x[i])**2
        boundary_pen += max(0, x[i] + r[i] - 1.0)**2
        boundary_pen += max(0, r[i] - y[i])**2
        boundary_pen += max(0, y[i] + r[i] - 1.0)**2
    
    # Overlap violations: circles must have disjoint interiors
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist:
                violation = min_dist - np.sqrt(dist_sq + 1e-12)
                overlap_pen += violation**2
    
    return boundary_pen + overlap_pen


def _objective(params):
    """Objective: maximize sum of radii (minimize negative sum + penalty)."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Penalty for constraint violations
    pen = _compute_penalty(x, y, r)
    
    # Maximize sum of radii => minimize -sum(r) + large_penalty*pen
    penalty_weight = 1000.0
    return -np.sum(r) + penalty_weight * pen


def _is_valid(x, y, r, tol=1e-7):
    """Check if configuration is valid."""
    # Boundary check
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
        if r[i] < -tol:
            return False
    
    # Overlap check
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < (min_dist - tol)**2:
                return False
    
    return True


def _grid_init(rows, cols, jitter=0.0):
    """Generate grid initialization."""
    params = np.zeros(3*N)
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x0 = (j + 0.5) / cols
            y0 = (i + 0.5) / rows
            r0 = 0.5 / max(rows, cols) - 0.01
            if jitter > 0:
                x0 += np.random.uniform(-jitter, jitter)
                y0 += np.random.uniform(-jitter, jitter)
            params[count] = x0
            params[N + count] = y0
            params[2*N + count] = r0
            count += 1
    # Fill remaining with small circles
    while count < N:
        params[count] = np.random.uniform(0.1, 0.9)
        params[N + count] = np.random.uniform(0.1, 0.9)
        params[2*N + count] = 0.02
        count += 1
    return params


def _cluster_init(n_clusters=4):
    """Generate clustered initialization."""
    params = np.zeros(3*N)
    centers = np.random.uniform(0.2, 0.8, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        x0 = centers[c, 0] + np.random.uniform(-0.1, 0.1)
        y0 = centers[c, 1] + np.random.uniform(-0.1, 0.1)
        r0 = 0.05
        params[i] = x0
        params[N + i] = y0
        params[2*N + i] = r0
    return params


def construct():
    """Pack 32 circles in unit square, maximizing sum of radii."""
    start_time = time.time()
    best_params = None
    best_score = -np.inf
    
    # Multiple initialization strategies
    inits = [
        _grid_init(6, 6, jitter=0.0),
        _grid_init(8, 4, jitter=0.0),
        _grid_init(4, 8, jitter=0.0),
        _grid_init(6, 6, jitter=0.05),
        _cluster_init(4),
        _cluster_init(6),
    ]
    
    # Add perturbed versions of best grid
    for _ in range(3):
        base = _grid_init(6, 6, jitter=0.0)
        perturbed = base + np.random.uniform(-0.02, 0.02, size=3*N)
        # Clamp radii to positive
        perturbed[2*N:] = np.maximum(perturbed[2*N:], 0.01)
        inits.append(perturbed)
    
    for init_params in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try to optimize from this initialization
        try:
            result = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                x = result.x[:N]
                y = result.x[N:2*N]
                r = result.x[2*N:3*N]
                
                # Check validity
                if _is_valid(x, y, r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_params = result.x.copy()
        except Exception:
            continue
    
    # Polish phase: high-precision optimization from best found
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x = result.x[:N]
            y = result.x[N:2*N]
            r = result.x[2*N:3*N]
            if _is_valid(x, y, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Final perturbation-restart cycles
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        for _ in range(5):
            if time.time() - start_time > DEADLINE:
                break
            try:
                perturbed = best_params + np.random.uniform(-0.005, 0.005, size=3*N)
                perturbed[2*N:] = np.maximum(perturbed[2*N:], 0.005)
                
                result = minimize(
                    _objective,
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                x = result.x[:N]
                y = result.x[N:2*N]
                r = result.x[2*N:3*N]
                
                if _is_valid(x, y, r):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_params = result.x.copy()
            except Exception:
                continue
    
    # If no valid solution found, fall back to simple grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    return np.column_stack([x, y, r])
