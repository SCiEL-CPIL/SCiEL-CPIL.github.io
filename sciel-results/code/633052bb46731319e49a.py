"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity_check(circles, n=N):
    """Check if circles are valid: inside unit square, non-overlapping."""
    if len(circles) != n:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if r < 0 or x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist = ri + rj
            if dist_sq < min_dist**2 - 1e-10:
                return False
    return True

def _objective(params, n=N):
    """Minimize negative sum of radii subject to constraints."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Penalize if circles go outside unit square
    penalty = 0.0
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2
        if r[i] < 0:
            penalty += r[i]**2
    
    # Penalize overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist = r[i] + r[j]
            if dist_sq < min_dist**2:
                penalty += (min_dist**2 - dist_sq)
    
    return -np.sum(r) + 100.0 * penalty

def _initial_grid(n=N):
    """Create an initial grid arrangement."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    return np.array(cells[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n = N
    best_circles = _initial_grid(n)
    best_score = np.sum(best_circles[:, 2])
    
    # Try multiple random restarts with gradient-based optimization
    rng = np.random.RandomState(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with grid plus some randomness
        if restart == 0:
            circles = _initial_grid(n)
        else:
            # Random initialization
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i, 0] = rng.uniform(0.1, 0.9)
                circles[i, 1] = rng.uniform(0.1, 0.9)
                circles[i, 2] = rng.uniform(0.01, 0.08)
        
        # Flatten for optimization
        params = circles.flatten()
        
        # Optimize
        try:
            result = minimize(_objective, params, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if result.success or result.fun < -best_score + 100 * 0:  # Check if better
                new_params = result.x
                new_circles = new_params.reshape(n, 3)
                
                # Ensure validity
                if _validity_check(new_circles, n):
                    new_score = np.sum(new_circles[:, 2])
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = new_circles
        except Exception:
            continue
        
        # If we have a good solution, try local refinement with smaller perturbations
        if best_score > 1.2 and time.time() < deadline - 30:
            # Perturb current best and re-optimize
            perturbed = best_circles.copy()
            for i in range(n):
                perturbed[i, 0] += rng.normal(0, 0.01)
                perturbed[i, 1] += rng.normal(0, 0.01)
                perturbed[i, 2] += rng.normal(0, 0.005)
                perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, 0.99)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, 0.99)
                perturbed[i, 2] = np.clip(perturbed[i, 2], 0.005, 0.15)
            
            params = perturbed.flatten()
            try:
                result = minimize(_objective, params, method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                new_params = result.x
                new_circles = new_params.reshape(n, 3)
                if _validity_check(new_circles, n):
                    new_score = np.sum(new_circles[:, 2])
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = new_circles
            except Exception:
                continue
    
    # Final validation
    if not _validity_check(best_circles, n):
        # Fall back to grid
        best_circles = _initial_grid(n)
    
    return best_circles
# EVOLVE-BLOCK-END
