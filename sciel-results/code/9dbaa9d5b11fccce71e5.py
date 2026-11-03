"""Improved circle packing in unit square using physics-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _initial_grid(n):
    """Create a grid layout as initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions = positions[:n]
    # Initial radius estimate
    r0 = 0.5 / side * 0.9
    return np.array(positions), r0


def _penalty_and_objective(params, n, penalty_weight=100.0):
    """
    Objective function: minimize negative sum of radii + penalties for constraint violations.
    params: array of shape (n*3,) -> [x0, y0, r0, x1, y1, r1, ...]
    """
    params = np.clip(params, 0.0, 1.0)  # Keep within bounds
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Objective: maximize sum of radii -> minimize negative sum
    obj = -np.sum(radii)
    
    # Penalty for circles going outside [0,1]^2
    # For each circle: x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    violations = 0.0
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if x - r < 0:
            violations += (x - r) ** 2
        if y - r < 0:
            violations += (y - r) ** 2
        if x + r > 1:
            violations += (x + r - 1) ** 2
        if y + r > 1:
            violations += (y + r - 1) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                violations += (min_dist - dist) ** 2
    
    return obj + penalty_weight * violations


def _is_valid(params, n, tol=1e-6):
    """Check if configuration is valid (no overlaps, within bounds)."""
    params = np.clip(params, 0.0, 1.0)
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Check bounds
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True


def _fix_to_boundary(params, n):
    """Fix any slight boundary violations by adjusting radii down."""
    params = np.array(params, dtype=float)
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        # Adjust radius to fit within bounds
        max_r = min(x, y, 1-x, 1-y)
        if r > max_r:
            radii[i] = max(0.0, max_r)
    
    # Fix overlaps by reducing radii
    for _ in range(10):  # Iterate a few times
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Reduce both radii
                    excess = min_dist - dist
                    radii[i] -= excess / 2
                    radii[j] -= excess / 2
                    changed = True
        if not changed:
            break
    
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    result = np.zeros(3*n)
    result[:2*n] = positions.flatten()
    result[2*n:] = radii
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple random restarts
    n_restarts = 0
    while time.time() < deadline - 5.0:  # Leave 5s safety margin
        n_restarts += 1
        
        # Generate initial guess
        np.random.seed(42 + n_restarts)
        positions, r0 = _initial_grid(N)
        
        # Add some random perturbation
        if n_restarts > 1:
            positions += np.random.uniform(-0.05, 0.05, size=positions.shape)
            positions = np.clip(positions, 0.0, 1.0)
        
        # Initial parameters
        params0 = np.zeros(3*N)
        params0[:2*N] = positions.flatten()
        params0[2*N:] = r0
        
        # Optimize
        try:
            result = minimize(
                _penalty_and_objective,
                params0,
                args=(N,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (3*N),
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                # Fix any minor violations
                fixed_params = _fix_to_boundary(result.x, N)
                
                # Check validity
                if _is_valid(fixed_params, N):
                    score = np.sum(fixed_params[2*N:])
                    if score > best_score:
                        best_score = score
                        best_result = fixed_params.reshape(N, 3)
                        
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # If no valid result found, return the grid seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
