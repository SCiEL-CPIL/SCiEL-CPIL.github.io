"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _is_valid(circles, eps=1e-9):
    """Check if all circles are inside [0,1]^2 and disjoint."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < -eps or y - r < -eps or x + r > 1 + eps or y + r > 1 + eps:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True


def _objective(params, n):
    """Penalty-based objective: maximize sum of radii, penalize violations."""
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Penalty for going outside unit square
    penalty = 0.0
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return -np.sum(r) + 1000.0 * penalty


def _local_optimize(circles, n, max_iter=5000):
    """Use scipy minimize to locally optimize a packing."""
    x0 = circles.flatten()
    res = minimize(_objective, x0, method='L-BFGS-B', 
                   bounds=[(0, 1)] * n + [(0, 1)] * n + [(0.01, 0.25)] * n,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.success or res.fun < _objective(x0, n):
        new_circles = res.x.reshape(n, 3)
        # Clip to valid range
        new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
        new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
        new_circles[:, 2] = np.clip(new_circles[:, 2], 0.001, 0.25)
        if _is_valid(new_circles):
            return new_circles
    return circles


def _perturb(circles, n, strength=0.05):
    """Add random perturbation to create diversity."""
    new_circles = circles.copy()
    for i in range(n):
        new_circles[i, 0] += random.gauss(0, strength)
        new_circles[i, 1] += random.gauss(0, strength)
        new_circles[i, 2] += random.gauss(0, strength * 0.5)
    # Clip
    new_circles[:, 0] = np.clip(new_circles[:, 0], 0.01, 0.99)
    new_circles[:, 1] = np.clip(new_circles[:, 1], 0.01, 0.99)
    new_circles[:, 2] = np.clip(new_circles[:, 2], 0.005, 0.25)
    return new_circles


def _initial_grid(n):
    """Create a reasonable initial grid packing."""
    # Use a hexagonal-like pattern for better initial packing
    side = int(np.ceil(np.sqrt(n)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= n:
                break
            # Offset every other row for hexagonal packing
            x_off = 0.5 / side if i % 2 == 0 else 0.5 / side + 0.25 / side
            x = (i + x_off) / side
            y = (j + 0.5) / side
            r = 0.45 / side
            circles.append([x, y, r])
    return np.array(circles[:n])


def _initial_random(n):
    """Create a random initial packing with non-overlapping circles."""
    circles = []
    random.seed(42)
    max_attempts = 1000
    for _ in range(max_attempts):
        if len(circles) >= n:
            break
        r = random.uniform(0.02, 0.15)
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        # Check overlap with existing circles
        valid = True
        for cx, cy, cr in circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r + cr - 1e-9:
                valid = False
                break
        if valid:
            circles.append([x, y, r])
    # If we didn't get enough, add smaller circles
    while len(circles) < n:
        r = 0.01
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        valid = True
        for cx, cy, cr in circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r + cr - 1e-9:
                valid = False
                break
        if valid:
            circles.append([x, y, r])
        else:
            # Try a different position
            x = random.uniform(r, 1 - r)
            y = random.uniform(r, 1 - r)
    return np.array(circles[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_circles = None
    best_score = -np.inf
    
    # Multiple strategies
    strategies = [
        ('grid', _initial_grid),
        ('random', _initial_random),
    ]
    
    random.seed(123)
    np.random.seed(123)
    
    for strat_name, init_func in strategies:
        if time.time() > deadline - 30:
            break
        
        # Try multiple restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            circles = init_func(N)
            if circles.shape[0] < N:
                continue
            
            # Local optimization
            circles = _local_optimize(circles, N, max_iter=3000)
            
            if _is_valid(circles):
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
            
            # Perturb and re-optimize
            if time.time() > deadline - 15:
                break
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                perturbed = _perturb(circles, N, strength=0.03)
                circles = _local_optimize(perturbed, N, max_iter=2000)
                if _is_valid(circles):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
    
    # Final refinement if time permits
    if best_circles is not None and time.time() < deadline - 5:
        best_circles = _local_optimize(best_circles, N, max_iter=5000)
        if not _is_valid(best_circles):
            # Revert to previous best
            pass
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
