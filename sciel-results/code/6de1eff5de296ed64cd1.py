# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hexagonal lattice prior and constrained refinement to equalize critical distances.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def objective(params):
        """
        Objective function to minimize.
        We want to maximize (d_min / d_max)^2.
        Equivalent to minimizing -(d_min / d_max)^2.
        
        To make the optimization more stable and gradient-friendly, 
        we can also think of this as minimizing the spread of distances 
        relative to the mean, but the direct ratio is the metric.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
        
        ratio = d_min / d_max
        # We minimize the negative square of the ratio
        return -(ratio ** 2)

    def evaluate(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -1.0

    def update_best(points):
        nonlocal best_score, best_points
        score = evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Strategy:
    # 1. Generate a large hexagonal lattice.
    # 2. Identify all subsets of 16 points from this lattice that are "compact" 
    #    (i.e., have a small bounding box relative to their min distance).
    #    Actually, it's better to just pick the 16 points closest to the origin 
    #    from a large hex lattice, as this is known to be a good packing.
    # 3. Refine these points using Nelder-Mead.
    
    # Generate Hexagonal Lattice
    # We need enough points to choose 16 from. 
    # A hex lattice with radius R has approx 3R^2 points.
    # Let's generate a lattice with R=5 (approx 75 points) to have plenty of choice.
    
    def generate_hex_lattice(R):
        points = []
        for q in range(-R, R + 1):
            for r in range(max(-R, -q - R), min(R, -q + R) + 1):
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                points.append([x, y])
        return np.array(points)

    # We will try a few different "center" configurations or just the densest 16.
    # The densest 16 points in a hex lattice centered at 0 are the best starting guess.
    
    lattice_R = 5
    hex_points = generate_hex_lattice(lattice_R)
    
    # Center the lattice
    hex_points -= np.mean(hex_points, axis=0)
    
    # Sort by distance from center to get the most compact cluster
    dists_center = np.linalg.norm(hex_points, axis=1)
    sorted_idx = np.argsort(dists_center)
    
    # Primary Start: The 16 closest to the center
    start_points = hex_points[sorted_idx[:16]].copy()
    
    # Normalize to unit scale to help optimizer
    norm = np.max(np.linalg.norm(start_points, axis=1))
    if norm > 1e-10:
        start_points = start_points / norm

    # Refine with Nelder-Mead
    # Nelder-Mead is derivative-free and robust for this non-smooth objective (min/max are non-differentiable at ties)
    
    if not time_exceeded():
        try:
            res = minimize(objective, start_points.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12},
                           callback=lambda x: time_exceeded())
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # Secondary Start: Try a slightly different subset or a square grid if hex fails or to compare
    # Square grid is a strong competitor for N=16 (4x4)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_start = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Normalize grid
    norm_g = np.max(np.linalg.norm(grid_start, axis=1))
    if norm_g > 1e-10:
        grid_start = grid_start / norm_g

    if not time_exceeded():
        try:
            res = minimize(objective, grid_start.ravel(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12},
                           callback=lambda x: time_exceeded())
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # Tertiary Start: Random perturbation of the best found so far
    if best_points is not None and not time_exceeded():
        for i in range(5):
            if time_exceeded(): break
            np.random.seed(123 + i)
            noise = np.random.randn(n, 2) * 0.05
            perturbed = best_points + noise
            norm_p = np.max(np.linalg.norm(perturbed, axis=1))
            if norm_p > 1e-10:
                perturbed = perturbed / norm_p
            
            try:
                res = minimize(objective, perturbed.ravel(), method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12},
                               callback=lambda x: time_exceeded())
                if not time_exceeded():
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Fallback
    if best_points is None:
        best_points = start_points

    return best_points

# EVOLVE-BLOCK-END
