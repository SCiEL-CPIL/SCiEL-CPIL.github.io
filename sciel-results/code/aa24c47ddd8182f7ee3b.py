"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _validity_penalty(x, n):
    """Compute a penalty for constraint violations (circles out of bounds or overlapping)."""
    # x is flattened: [x0, y0, r0, x1, y1, r1, ...]
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    penalty = 0.0
    
    # Boundary constraints: circle must be inside [0,1]^2
    # x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    # We penalize violations
    for i in range(n):
        r = radii[i]
        cx, cy = centers[i]
        # If circle goes out of bounds, add penalty proportional to violation
        if cx - r < 0:
            penalty += (cx - r) ** 2
        if cy - r < 0:
            penalty += (cy - r) ** 2
        if cx + r > 1:
            penalty += (cx + r - 1) ** 2
        if cy + r > 1:
            penalty += (cy + r - 1) ** 2
    
    # Overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return penalty


def _objective(x, n):
    """Objective: maximize sum of radii, so minimize negative sum of radii + penalty."""
    radii = x[2 * n:]
    sum_radii = np.sum(radii)
    penalty = _validity_penalty(x, n)
    # We want to maximize sum_radii, so minimize -sum_radii + penalty
    # But penalty should dominate to ensure feasibility
    return -sum_radii + 100.0 * penalty


def _gradient(x, n):
    """Gradient of the objective."""
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    grad = np.zeros_like(x)
    
    # Gradient of -sum(radii)
    grad[2 * n:] = -1.0
    
    # Gradient of penalty
    # Boundary penalties
    for i in range(n):
        r = radii[i]
        cx, cy = centers[i]
        # cx - r < 0: penalty = (cx - r)^2, d/d(cx) = 2(cx-r), d/d(r) = -2(cx-r)
        if cx - r < 0:
            grad[2 * i] += 2.0 * (cx - r)
            grad[2 * i + 1] += 0
            grad[2 * n + i] += -2.0 * (cx - r)
        if cy - r < 0:
            grad[2 * i] += 0
            grad[2 * i + 1] += 2.0 * (cy - r)
            grad[2 * n + i] += -2.0 * (cy - r)
        if cx + r > 1:
            grad[2 * i] += 2.0 * (cx + r - 1)
            grad[2 * i + 1] += 0
            grad[2 * n + i] += 2.0 * (cx + r - 1)
        if cy + r > 1:
            grad[2 * i] += 0
            grad[2 * i + 1] += 2.0 * (cy + r - 1)
            grad[2 * n + i] += 2.0 * (cy + r - 1)
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            min_dist = radii[i] + radii[j]
            if dist < min_dist and dist > 1e-10:
                # penalty = (min_dist - dist)^2
                # d/d(centers[i]) = -2*(min_dist - dist) * (diff/dist)
                # d/d(centers[j]) = +2*(min_dist - dist) * (diff/dist)
                # d/d(radii[i]) = -2*(min_dist - dist)
                # d/d(radii[j]) = -2*(min_dist - dist)
                factor = 2.0 * (min_dist - dist) / dist
                grad[2 * i:2 * i + 2] += -factor * diff
                grad[2 * j:2 * j + 2] += factor * diff
                grad[2 * n + i] += -2.0 * (min_dist - dist)
                grad[2 * n + j] += -2.0 * (min_dist - dist)
    
    return grad


def _make_initial_grid(n):
    """Create initial grid configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            cells.append([x, y, r])
    return np.array(cells[:n])


def _make_initial_random(n, seed):
    """Create random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.02, 0.08, size=n)
    return np.hstack([centers, radii.reshape(n, 1)]).flatten()


def construct():
    start_time = time.time()
    best_x = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # Grid initialization
    grid = _make_initial_grid(N)
    inits.append(grid.flatten())
    
    # Random initializations
    for seed in [42, 123, 456, 789, 101, 202]:
        inits.append(_make_initial_random(N, seed))
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        x0 = init.copy()
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.49))  # x
            bounds.append((0.01, 0.49))  # y
            bounds.append((0.01, 0.49))  # r
        
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                jac=_gradient,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                x = res.x
                radii = x[2 * N:]
                score = np.sum(radii)
                
                # Check validity
                centers = x[:2 * N].reshape(N, 2)
                valid = True
                for i in range(N):
                    r = radii[i]
                    cx, cy = centers[i]
                    if cx - r < -1e-6 or cy - r < -1e-6 or cx + r > 1 + 1e-6 or cy + r > 1 + 1e-6:
                        valid = False
                        break
                    for j in range(i + 1, N):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < radii[i] + radii[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    if score > best_score:
                        best_score = score
                        best_x = x.copy()
                        
        except Exception:
            continue
    
    # If no valid solution found, use the grid as fallback
    if best_x is None:
        grid = _make_initial_grid(N)
        best_x = grid.flatten()
    
    # Extract result
    centers = best_x[:2 * N].reshape(N, 2)
    radii = best_x[2 * N:]
    result = np.hstack([centers, radii.reshape(N, 1)])
    
    return result
