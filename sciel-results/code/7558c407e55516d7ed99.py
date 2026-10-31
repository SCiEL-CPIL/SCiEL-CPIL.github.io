"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _compute_score(circles):
    """
    Compute the objective value: sum of radii, but penalize if any constraint is violated.
    Constraints:
    1. All circles inside [0,1]^2: r <= x <= 1-r, r <= y <= 1-r
    2. Disjoint interiors: dist(i,j) >= r_i + r_j for all i != j
    """
    n = len(circles)
    if n == 0:
        return 0.0
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(r < 0) or np.any(x < r) or np.any(y < r) or np.any(x > 1 - r) or np.any(y > 1 - r):
        # Heavy penalty for bound violations
        penalty = 1e6
        return -penalty
    
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                # Penalty proportional to violation
                violation = (r[i] + r[j]) - dist
                return -(1e6 + violation * 1e6)
    
    return np.sum(r)

def _is_valid(circles):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r < 1e-9):
        return False
    if np.any(x < r - 1e-9) or np.any(y < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y > 1 - r + 1e-9):
        return False
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _gradient_ascent_step(circles, lr, n_iter=500):
    """
    Perform gradient ascent on the sum of radii, using subgradients of the constraints.
    We use a simple iterative approach: for each circle, try to increase radius slightly,
    and adjust positions to maintain feasibility.
    """
    n = len(circles)
    current = circles.copy()
    
    for it in range(n_iter):
        improved = False
        for i in range(n):
            # Try to increase radius of circle i
            new_r = current[i, 2] * 1.001 + 1e-6
            if new_r >= 0.5:
                continue
            
            # Check if we can increase radius while staying in bounds
            x, y, r = current[i]
            if new_r > min(x, 1-x, y, 1-y):
                continue
            
            # Check pairwise constraints with new radius
            feasible = True
            for j in range(n):
                if i == j:
                    continue
                dx = x - current[j, 0]
                dy = y - current[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_r + current[j, 2] - 1e-9:
                    feasible = False
                    break
            
            if feasible:
                current[i, 2] = new_r
                improved = True
        
        if not improved:
            break
    
    return current

def _local_optimization(circles, max_iters=1000, tol=1e-8):
    """
    Local optimization using gradient-based methods.
    We maximize sum(r_i) subject to constraints.
    Use a penalty method or barrier method.
    """
    n = len(circles)
    current = circles.copy()
    
    # Use a simple iterative scheme: for each circle, try to expand it
    # while pushing away overlapping neighbors.
    
    for outer in range(max_iters):
        changed = False
        for i in range(n):
            xi, yi, ri = current[i]
            
            # Max possible radius given bounds
            max_r_bounds = min(xi, 1-xi, yi, 1-yi)
            
            # Max radius given neighbors
            max_r_neighbors = max_r_bounds
            for j in range(n):
                if i == j:
                    continue
                dx = xi - current[j, 0]
                dy = yi - current[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # dist >= ri + rj => ri <= dist - rj
                candidate = dist - current[j, 2]
                if candidate < max_r_neighbors:
                    max_r_neighbors = candidate
            
            max_r_neighbors = max(0, max_r_neighbors)
            
            # If current radius is less than max possible, increase it
            if ri < max_r_neighbors - 1e-10:
                # Increase radius to max possible
                current[i, 2] = max_r_neighbors
                changed = True
            else:
                # Try to move the circle to allow larger radius
                # Find the most constraining neighbor
                min_slack = np.inf
                constraining_j = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = xi - current[j, 0]
                    dy = yi - current[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    slack = dist - ri - current[j, 2]
                    if slack < min_slack:
                        min_slack = slack
                        constraining_j = j
                
                if constraining_j >= 0 and min_slack < 1e-6:
                    # Move away from constraining neighbor
                    dx = xi - current[constraining_j, 0]
                    dy = yi - current[constraining_j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-9:
                        # Move in direction away from neighbor
                        step = 0.01
                        new_x = xi + dx/dist * step
                        new_y = yi + dy/dist * step
                        # Check bounds
                        if ri <= new_x <= 1-ri and ri <= new_y <= 1-ri:
                            # Check all other constraints
                            feasible = True
                            for j in range(n):
                                if i == j:
                                    continue
                                dx2 = new_x - current[j, 0]
                                dy2 = new_y - current[j, 1]
                                dist2 = np.sqrt(dx2*dx2 + dy2*dy2)
                                if dist2 < ri + current[j, 2] - 1e-9:
                                    feasible = False
                                    break
                            if feasible:
                                current[i, 0] = new_x
                                current[i, 1] = new_y
                                changed = True
        
        if not changed:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    random.seed(42)
    
    # Start with a reasonable initial configuration
    # Use a grid-like pattern but with some randomness
    side = 5  # 5x5 grid gives 25 cells, we need 26
    # Use 5x5 grid, put 25 circles, then add one more
    
    best_circles = None
    best_score = -np.inf
    
    # Initial configuration: 5x5 grid
    circles = np.zeros((N, 3))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= N:
                break
            circles[idx, 0] = (i + 0.5) / 5
            circles[idx, 1] = (j + 0.5) / 5
            circles[idx, 2] = 0.5 / 5 - 0.01  # Slightly smaller to ensure feasibility
            idx += 1
    
    # For the 26th circle, place it in a corner or somewhere
    # Actually, let's use a different approach: 5x5 grid with 25 circles, 
    # and the 26th circle smaller, placed in a gap
    
    # Let's try a cleaner initial setup
    # Place circles in a 5x5 grid, radius = 0.08
    r_init = 0.08
    circles = np.zeros((N, 3))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= N:
                break
            circles[idx, 0] = (i + 0.5) / 5
            circles[idx, 1] = (j + 0.5) / 5
            circles[idx, 2] = r_init
            idx += 1
    
    # The 26th circle - place it in a corner
    circles[25, 0] = r_init + 0.01
    circles[25, 1] = r_init + 0.01
    circles[25, 2] = r_init * 0.5
    
    # Verify initial validity
    if _is_valid(circles):
        best_circles = circles.copy()
        best_score = np.sum(circles[:, 2])
    
    # Iterative local optimization
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Local optimization
        optimized = _local_optimization(circles, max_iters=200)
        
        if _is_valid(optimized):
            score = np.sum(optimized[:, 2])
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
                circles = optimized.copy()
        
        # Try a random perturbation to escape local optima
        if time.time() < deadline - 5:
            perturbed = circles.copy()
            # Perturb a few random circles
            for _ in range(random.randint(1, 3)):
                k = random.randint(0, N-1)
                dx = random.uniform(-0.02, 0.02)
                dy = random.uniform(-0.02, 0.02)
                new_x = perturbed[k, 0] + dx
                new_y = perturbed[k, 1] + dy
                if perturbed[k, 2] <= new_x <= 1-perturbed[k, 2] and \
                   perturbed[k, 2] <= new_y <= 1-perturbed[k, 2]:
                    perturbed[k, 0] = new_x
                    perturbed[k, 1] = new_y
            
            if _is_valid(perturbed):
                # Optimize from perturbed position
                opt_perturbed = _local_optimization(perturbed, max_iters=200)
                if _is_valid(opt_perturbed):
                    score = np.sum(opt_perturbed[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = opt_perturbed.copy()
                        circles = opt_perturbed.copy()
    
    # Final high-precision polish
    if best_circles is not None and time.time() < deadline:
        polished = _local_optimization(best_circles, max_iters=5000, tol=1e-10)
        if _is_valid(polished):
            best_circles = polished
    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
