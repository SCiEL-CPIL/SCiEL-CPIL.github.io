"""
Strategy: Simulated Annealing with geometric initialization and adaptive cooling.
This replaces force-directed relaxation with a stochastic global search method that
can escape local minima by accepting worse solutions with decreasing probability.
"""

import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 118.0  # Safety margin before 120s limit

def compute_radius(points):
    """Compute the maximum common radius for a set of points."""
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        points[:, 0] + 0.5,
        0.5 - points[:, 0],
        points[:, 1] + 0.5,
        0.5 - points[:, 1],
        points[:, 2] + 0.5,
        0.5 - points[:, 2],
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def init_grid(n):
    """
    Initialize points in a simple cubic grid pattern.
    This provides a structured starting point that is easy to understand and modify.
    """
    points = []
    # Target radius is approx 0.111. Diameter 0.222.
    # In a cube of side 1, we can fit roughly (1/0.222)^3 ~ 90 spheres in simple cubic.
    # Let's use a grid with spacing ~ 0.2
    
    s = 0.2
    grid_range = np.arange(-0.5, 0.5, s)
    
    for x in grid_range:
        for y in grid_range:
            for z in grid_range:
                p = np.array([x, y, z])
                # Check if inside cube
                if np.all(p >= -0.5) and np.all(p <= 0.5):
                    points.append(p)
    
    points = np.array(points)
    
    # If we have more than N, select N using farthest-point sampling
    if len(points) > n:
        selected = [points[0]]
        remaining = points[1:]
        
        for _ in range(n - 1):
            min_dists = np.array([np.min(np.linalg.norm(p - np.array(selected), axis=1)) for p in remaining])
            best_idx = np.argmax(min_dists)
            selected.append(remaining[best_idx])
            remaining = np.delete(remaining, best_idx, axis=0)
            
        points = np.array(selected)
        
    elif len(points) < n:
        # Pad with random points if grid is too sparse
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(points), 3))
        points = np.vstack([points, extra])
        
    return points[:n]

def simulated_annealing(points, time_budget, seed=42):
    """
    Simulated Annealing:
    - Start with high temperature
    - At each step, propose a random move (shift one point slightly)
    - Accept with probability exp(-delta_E / T) where delta_E is the change in objective
    - Gradually decrease temperature
    """
    np.random.seed(seed)
    pts = points.copy()
    start_time = time.time()
    
    r_cur = compute_radius(pts)
    best_pts = pts.copy()
    best_r = r_cur
    
    # Temperature schedule
    T_start = 0.01
    T_end = 1e-6
    cooling_rate = 0.9995
    
    T = T_start
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        # Propose a move: pick a random point and shift it
        i = np.random.randint(len(pts))
        delta = np.random.randn(3) * step_size
        new_pts = pts.copy()
        new_pts[i] += delta
        # Clip to cube
        new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
        
        r_new = compute_radius(new_pts)
        delta_r = r_new - r_cur
        
        # Acceptance criterion
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            pts = new_pts
            r_cur = r_new
            
            # Update best
            if r_cur > best_r:
                best_r = r_cur
                best_pts = pts.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # Adapt step size
        # If we're making progress, keep step size; if stuck, reduce it
        # Simple heuristic: reduce step size as temperature drops
        step_size = 0.01 * (T / T_start) + 0.001
        
        # Check for convergence
        if T < 1e-5:
            break
    
    return best_pts, best_r

def hill_climbing(points, time_budget, seed=42):
    """
    Simple hill climbing: try small moves and accept only if they improve the radius.
    """
    np.random.seed(seed)
    pts = points.copy()
    start_time = time.time()
    
    r_cur = compute_radius(pts)
    step = 0.005
    
    improved = True
    while improved and (time.time() - start_time < time_budget):
        improved = False
        for i in range(len(pts)):
            for axis in range(3):
                for direction in [-1, 1]:
                    new_pts = pts.copy()
                    new_pts[i, axis] += direction * step
                    new_pts[i, axis] = np.clip(new_pts[i, axis], -0.5, 0.5)
                    
                    r_new = compute_radius(new_pts)
                    if r_new > r_cur:
                        pts = new_pts
                        r_cur = r_new
                        improved = True
        
        # Reduce step size if no improvement
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
                
    return pts, r_cur

def construct():
    start_time = time.time()
    
    # Phase 1: Grid Initialization
    points = init_grid(N)
    best_pts = points.copy()
    best_r = compute_radius(points)
    
    # Phase 2: Simulated Annealing
    # Budget: 80 seconds
    sa_budget = min(80.0, DEADLINE - (time.time() - start_time) - 15)
    if sa_budget > 5:
        pts1, r1 = simulated_annealing(points, sa_budget, seed=42)
        if r1 > best_r:
            best_r = r1
            best_pts = pts1
    
    # Phase 3: Hill Climbing Polish
    # Budget: remaining time - 5s safety
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 5:
        pts2, r2 = hill_climbing(best_pts, remaining, seed=123)
        if r2 > best_r:
            best_r = r2
            best_pts = pts2
    
    # Final check and return
    result = np.clip(best_pts, -0.5, 0.5)
    final_r = compute_radius(result)
    
    # Ensure validity
    if not np.all(np.isfinite(result)):
        # Fallback to grid
        return init_grid(N)
        
    return result
