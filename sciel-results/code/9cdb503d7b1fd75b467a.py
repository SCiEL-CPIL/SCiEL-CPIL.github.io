"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softmin(v, beta):
    """Smooth approximation of min(v) using log-sum-exp."""
    m = np.min(v)
    return m - np.log(np.sum(np.exp(beta * (v - m)))) / beta

def _hex_initialization(n):
    """Create a hexagonal close-packing layout for n circles in unit square."""
    positions = []
    # Hexagonal grid with spacing optimized for unit square
    # For hexagonal packing, row spacing is r*sqrt(3), column spacing is 2*r
    # We'll use a parameter to scale
    scale = 1.0
    row = 0
    while len(positions) < n:
        cols_in_row = int(np.floor((1.0 + scale * 0.5 * (row % 2)) / (scale * 1.0)))
        for col in range(cols_in_row):
            if len(positions) >= n:
                break
            x = scale * (0.5 + col + 0.5 * (row % 2))
            y = scale * (0.5 + row * np.sqrt(3) / 2)
            positions.append([x, y])
        row += 1
    
    # Normalize to fit in unit square
    positions = np.array(positions[:n])
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    # Scale to fit with margin
    margin = 0.05
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    scale_factor = min((1 - 2*margin) / span_x, (1 - 2*margin) / span_y)
    positions = (positions - np.array([xmin, ymin])) * scale_factor + margin
    return positions

def _objective(params, n, beta):
    """Objective: negative sum of radii with smooth constraint penalty."""
    x_coords = params[0:2*n:2]
    y_coords = params[1:2*n:2]
    r = params[2*n:]
    
    # Boundary constraints
    v1 = np.maximum(0, r - x_coords)
    v2 = np.maximum(0, x_coords + r - 1)
    v3 = np.maximum(0, r - y_coords)
    v4 = np.maximum(0, y_coords + r - 1)
    boundary_violations = np.concatenate([v1, v2, v3, v4])
    
    # Overlap constraints
    overlap_violations = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x_coords[i] - x_coords[j]
            dy = y_coords[i] - y_coords[j]
            dist = np.sqrt(dx*dx + dy*dy)
            violation = np.maximum(0, r[i] + r[j] - dist)
            overlap_violations.append(violation)
    overlap_violations = np.array(overlap_violations)
    
    # Smooth penalty using soft-min surrogate
    penalty = _softmin(boundary_violations, beta) * 100 + _softmin(overlap_violations, beta) * 100
    
    return -np.sum(r) + penalty

def _is_valid(result, tol=1e-6):
    """Check if packing is valid."""
    n = result.shape[0]
    for i in range(n):
        x, y, r = result[i]
        if r <= 0 or x < r - tol or y < r - tol or x > 1 - r + tol or y > 1 - r + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(result[i,:2] - result[j,:2])
            if dist < result[i,2] + result[j,2] - tol:
                return False
    return True

def _clean(result, tol=1e-8):
    """Clean up numerical issues."""
    result = result.copy()
    for i in range(result.shape[0]):
        x, y, r = result[i]
        r = max(r, 1e-10)
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        result[i] = [x, y, r]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple initializations and optimizations
    n = N
    
    # Hexagonal initialization
    positions = _hex_initialization(n)
    
    # Initial radii: estimate from packing density
    # Hexagonal packing density ~ 0.9069
    # Total area = sum(pi*r^2) <= 0.9069 * 1
    # If all equal: n*pi*r^2 = 0.9069 -> r = sqrt(0.9069/(n*pi))
    r0 = np.sqrt(0.9069 / (n * np.pi))
    
    # Start with slightly smaller radii
    r = np.full(n, r0 * 0.95)
    
    # Pack into parameter vector
    params = np.concatenate([positions.flatten(), r])
    
    # Optimize using L-BFGS-B with smooth objective
    beta_schedule = [10, 20, 50, 100, 200, 500]
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        res = minimize(
            _objective,
            params,
            args=(n, beta),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if res.success or res.fun < best_score:
            params = res.x
            # Extract and clean
            result = np.column_stack([
                params[:2*n].reshape(n, 2),
                params[2*n:]
            ])
            result = _clean(result)
            
            if _is_valid(result):
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
        
        if time.time() > deadline:
            break
    
    # If no valid result, fall back to grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
