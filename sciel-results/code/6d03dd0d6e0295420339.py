"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Structured 3x4 grid topology with rubber-band springs, optimized via
differential evolution and Nelder-Mead polishing on a soft-min objective.
The 11 points are arranged in a 3x4 grid (with one corner removed) to exploit
known structural regularity of optimal Heilbronn configurations. Each point is
connected to its grid neighbors by springs that pull it toward the average of
its neighbors, creating a quasi-lattice structure. The optimization variables
are the Cartesian coordinates of all 11 points, constrained to lie inside the
equilateral triangle via projection. The objective is a soft-min approximation
of the minimum triangle area, with a penalty term for points outside the triangle.
"""
import numpy as np
import time
import random
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Grid topology: 3 rows x 4 columns, with one point removed (11 points total)
# We use a 3x4 grid and remove the last point, or use a specific pattern.
# For 11 points, a natural arrangement is 3+4+4 or 4+3+4.
# Let's use a 3x4 grid with the bottom-right corner removed:
# Row 0: 4 points
# Row 1: 4 points  
# Row 2: 3 points
# Total: 11 points

GRID_ROWS = 3
GRID_COLS = 4
# Point indices in grid order (row-major):
# 0  1  2  3
# 4  5  6  7
# 8  9  10
# Point 11 would be at (2,3) but we only have 11 points (indices 0-10)

def _grid_neighbors(idx):
    """Return list of neighbor indices for a point in the 3x4 grid (last point removed)."""
    # Grid positions:
    # idx 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
    # idx 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
    # idx 8: (2,0), 9: (2,1), 10: (2,2)
    # idx 11 would be (2,3) but doesn't exist
    
    row = idx // 4
    col = idx % 4
    
    neighbors = []
    
    # Check all 4-directional neighbors
    # Up
    if row > 0:
        nidx = (row - 1) * 4 + col
        if nidx < N:
            neighbors.append(nidx)
    # Down
    if row < GRID_ROWS - 1:
        nidx = (row + 1) * 4 + col
        if nidx < N:
            neighbors.append(nidx)
    # Left
    if col > 0:
        nidx = row * 4 + (col - 1)
        if nidx < N:
            neighbors.append(nidx)
    # Right
    if col < GRID_COLS - 1:
        nidx = row * 4 + (col + 1)
        if nidx < N:
            neighbors.append(nidx)
    
    return neighbors

# Precompute neighbor lists
NEIGHBORS = [tuple(_grid_neighbors(i)) for i in range(N)]

def _barycentric_to_cartesian(bary_pts):
    """Convert barycentric coordinates to Cartesian coordinates."""
    n = len(bary_pts)
    cart_pts = np.zeros((n, 2))
    cart_pts[:, 0] = bary_pts[:, 1] + 0.5 * bary_pts[:, 2]
    cart_pts[:, 1] = bary_pts[:, 2] * S3H
    return cart_pts

def _cartesian_to_barycentric(cart_pts):
    """Convert Cartesian coordinates to barycentric coordinates."""
    n = len(cart_pts)
    bary_pts = np.zeros((n, 3))
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    bary_pts[:, 0] = u
    bary_pts[:, 1] = v
    bary_pts[:, 2] = w
    return bary_pts

def _is_inside_triangle(pts):
    """Check if all points are inside or on the boundary of the triangle."""
    bary = _cartesian_to_barycentric(pts)
    return np.all(bary >= -1e-9)

def _project_to_triangle(pts):
    """Project points to the triangle using barycentric coordinates."""
    bary = _cartesian_to_barycentric(pts)
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return _barycentric_to_cartesian(bary)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _soft_min_area(pts, alpha):
    """Compute soft-min approximation of the minimum triangle area."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-alpha * area))) / alpha
    # For numerical stability, shift by min
    a_min = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-alpha * (areas - a_min))))
    soft_min = a_min - log_sum / alpha
    return soft_min

def _rubber_band_penalty(pts, spring_k=1.0):
    """
    Compute rubber band penalty: sum over each point of the squared distance
    from its position to the average of its neighbors' positions.
    This encourages a smooth, quasi-lattice structure.
    """
    penalty = 0.0
    for i in range(N):
        nbrs = NEIGHBORS[i]
        if len(nbrs) == 0:
            continue
        avg_pos = np.mean(pts[nbrs], axis=0)
        diff = pts[i] - avg_pos
        penalty += np.dot(diff, diff)
    return spring_k * penalty

def _objective(pts, alpha, spring_k=0.01):
    """
    Full objective: soft-min area + rubber band penalty.
    Higher is better for area, so we negate it for minimization.
    """
    if not _is_inside_triangle(pts):
        # Heavy penalty for points outside
        bary = _cartesian_to_barycentric(pts)
        outside_penalty = np.sum(np.minimum(bary, 0) ** 2)
        return -1e6 + outside_penalty
    
    area = _soft_min_area(pts, alpha)
    band_pen = _rubber_band_penalty(pts, spring_k)
    # We want to maximize area and minimize band penalty
    return -area + band_pen

def _random_grid_init(rng, jitter=0.1):
    """
    Generate a random initial configuration based on the 3x4 grid topology.
    Points are placed at grid positions with random jitter.
    """
    pts = np.zeros((N, 2))
    for i in range(N):
        row = i // 4
        col = i % 4
        # Grid position in the triangle
        # Map grid (row, col) to barycentric coordinates
        # Row 0 (top): near vertex C
        # Row 2 (bottom): near base AB
        # Col 0 (left): near vertex A
        # Col 3 (right): near vertex B
        
        # Barycentric coordinates for grid position
        # u = weight on A (bottom-left), v = weight on B (bottom-right), w = weight on C (top)
        # For a 3x4 grid:
        # w = 1 - (row + 0.5) / GRID_ROWS  (decreases from top to bottom)
        # u = (col + 0.5) / GRID_COLS * (1 - w)  (increases from left to right within the row)
        # v = (1 - w) - u
        
        w = 1.0 - (row + 0.5) / GRID_ROWS
        # Distribute u and v along the base
        u_frac = (col + 0.5) / GRID_COLS
        u = u_frac * (1.0 - w)
        v = (1.0 - w) - u
        
        bary = np.array([u, v, w])
        cart = _barycentric_to_cartesian(bary.reshape(1, 3))[0]
        
        # Add random jitter
        cart += jitter * np.array([rng.gauss(0, 1), rng.gauss(0, 1)])
        pts[i] = cart
    
    # Project to triangle
    pts = _project_to_triangle(pts)
    return pts

def _differential_evolution(pts, rng, alpha, spring_k, max_iter=200, pop_size=20):
    """
    Simple differential evolution for the 22-dimensional problem (11 points x 2 coords).
    """
    best_pts = pts.copy()
    best_obj = _objective(best_pts, alpha, spring_k)
    
    # Initialize population
    pop = [best_pts.copy()]
    for _ in range(pop_size - 1):
        candidate = _random_grid_init(rng, jitter=0.15)
        pop.append(candidate)
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    for iteration in range(max_iter):
        new_pop = [best_pts.copy()]
        
        for i in range(1, pop_size):
            # Select three distinct random individuals
            while True:
                a, b, c = rng.sample(range(pop_size), 3)
                if a != b and b != c and a != c:
                    break
            
            # Differential mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            trial = pop[i].copy()
            for d in range(2 * N):
                if rng.random() < CR:
                    trial[d] = mutant[d]
            # Ensure at least one parameter is taken from mutant
            if not np.any(trial != pop[i]):
                trial[rng.randint(0, 2 * N - 1)] = mutant[rng.randint(0, 2 * N - 1)]
            
            # Reshape and project
            trial_pts = trial.reshape(N, 2)
            trial_pts = _project_to_triangle(trial_pts)
            
            trial_obj = _objective(trial_pts, alpha, spring_k)
            
            if trial_obj < pop[i].reshape(-1).sum() or True:
                # Compare objectives properly
                current_obj = _objective(pop[i], alpha, spring_k)
                if trial_obj < current_obj:
                    new_pop.append(trial_pts)
                    if trial_obj < best_obj:
                        best_obj = trial_obj
                        best_pts = trial_pts.copy()
                else:
                    new_pop.append(pop[i].copy())
            else:
                new_pop.append(pop[i].copy())
        
        pop = new_pop
    
    return best_pts, best_obj

def _nelder_mead_polish(pts, alpha, spring_k, max_iter=500):
    """
    Nelder-Mead optimization for polishing the solution.
    """
    from scipy.optimize import minimize
    
    def objective_flat(x):
        pts = x.reshape(N, 2)
        pts = _project_to_triangle(pts)
        return _objective(pts, alpha, spring_k)
    
    x0 = pts.reshape(-1)
    result = minimize(objective_flat, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_pts = _project_to_triangle(result.x.reshape(N, 2))
    return best_pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses structured grid topology with rubber-band springs, optimized via
    differential evolution and Nelder-Mead polishing.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = -float('inf')
    
    # Multi-stage optimization with decreasing alpha (annealing)
    # Start with large alpha (smooth landscape) and decrease to small alpha (sharp minimum)
    alpha_schedule = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    spring_k_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    # Initial configurations
    inits = []
    for _ in range(3):
        inits.append(_random_grid_init(rng, jitter=0.1))
    
    # Also try a more structured init with less jitter
    inits.append(_random_grid_init(rng, jitter=0.02))
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        current_pts = init_pts.copy()
        
        for alpha, spring_k in zip(alpha_schedule, spring_k_schedule):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Differential evolution phase
            de_time_budget = min(5.0, (DEADLINE - 5 - (time.time() - start_time)) / 2)
            if de_time_budget > 0.5:
                current_pts, de_obj = _differential_evolution(
                    current_pts, rng, alpha, spring_k, 
                    max_iter=min(100, int(de_time_budget * 20))
                )
            
            # Nelder-Mead polishing phase
            nm_time_budget = min(3.0, DEADLINE - 5 - (time.time() - start_time))
            if nm_time_budget > 0.5:
                current_pts = _nelder_mead_polish(
                    current_pts, alpha, spring_k,
                    max_iter=min(200, int(nm_time_budget * 50))
                )
            
            # Evaluate
            current_area = _min_triangle_area(current_pts)
            if current_area > best_area:
                best_area = current_area
                best_pts = current_pts.copy()
        
        # Final polish with very small alpha
        if time.time() - start_time < DEADLINE - 3:
            final_pts = _nelder_mead_polish(current_pts, 0.5, 0.001, max_iter=300)
            final_area = _min_triangle_area(final_pts)
            if final_area > best_area:
                best_area = final_area
                best_pts = final_pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _random_grid_init(rng, jitter=0.05)
    
    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts
