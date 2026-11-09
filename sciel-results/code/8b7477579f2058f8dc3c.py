# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s

    # Precompute indices for all triples of points
    # i < j < k
    idx = np.array(list((i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)))
    # idx shape: (286, 3)
    
    def min_triangle_area_vectorized(points):
        """
        Calculate the minimum triangle area among all triples of points using vectorization.
        points: (n, 2)
        """
        # Extract points for all triples
        # p1: points[idx[:, 0]], shape (286, 2)
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Vectorized:
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        # Check for degeneracy (area close to 0)
        # If any area is extremely small, return a very small number to penalize
        min_area = np.min(areas)
        
        # If points are collinear or nearly so, min_area will be ~0
        return min_area

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective(points):
        """
        Objective function to maximize: A_min / A_hull.
        We minimize the negative of this ratio.
        """
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        # Penalize if A_min is effectively 0 (collinear points)
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / A_hull

    # Generate diverse initial configurations
    rng = np.random.default_rng(seed=42)
    
    initial_configs = []
    
    # 1. Regular polygon (baseline)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Perturbed regular polygon
    for _ in range(5):
        perturbation = rng.normal(0, 0.1, (n, 2))
        initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]) + perturbation)
        
    # 3. Random points in unit square
    for _ in range(5):
        initial_configs.append(rng.random((n, 2)))
        
    # 4. "Clustered" random points - often leads to better local minima for Heilbronn
    # Start with a few points, add more near existing ones but with some spacing
    for _ in range(5):
        pts = []
        # First point
        pts.append(rng.random(2))
        for i in range(1, n):
            # Pick a random existing point and add a random offset
            base = pts[rng.integers(0, len(pts))]
            offset = rng.normal(0, 0.2, 2)
            new_pt = base + offset
            pts.append(new_pt)
        initial_configs.append(np.array(pts))
        
    # 5. Grid-like with jitter
    grid_x = [-1, -0.5, 0, 0.5, 1]
    grid_y = [-1, -0.5, 0, 0.5, 1]
    for _ in range(5):
        pts = []
        for i in range(n):
            x = grid_x[rng.integers(0, 5)] + rng.normal(0, 0.1)
            y = grid_y[rng.integers(0, 5)] + rng.normal(0, 0.1)
            pts.append([x, y])
        initial_configs.append(np.array(pts))

    best_points = None
    best_score = -float('inf')
    
    # Sort initial configs by their initial score to prioritize good starts? 
    # Or just run them all. Running all is safer to avoid missing a basin.
    
    # Optimization loop
    # We have ~600s. Each Nelder-Mead run with maxiter=2000 on 13 points (26 vars) 
    # with vectorized area calc should be fast. Let's estimate.
    # 286 triangles * 3 points * 2 coords. Vectorized ops are very fast.
    # ConvexHull is the bottleneck. O(n log n).
    # Let's try to fit as many runs as possible.
    
    # We will run optimization on all initial configs, but break if time runs out.
    
    for i, init_points in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Normalize/Center the points to help optimization
        # Centering helps with numerical stability
        center = np.mean(init_points, axis=0)
        centered = init_points - center
        # Scale to roughly unit magnitude
        norm = np.linalg.norm(centered)
        if norm > 1e-10:
            centered = centered / norm * 1.0 # Scale to radius 1
            
        try:
            result = minimize(
                objective,
                centered,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': 0.1 # Smaller initial simplex for finer search
                }
            )
            
            # Re-center and scale back? No, the ratio is scale invariant.
            # But we must ensure the result is valid.
            if result.fun < 0: # Negative objective means positive score
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.copy()
                    
        except Exception:
            pass

    # If optimization failed or didn't improve, check the initial configs directly
    if best_points is None:
        for init_points in initial_configs:
            # Center and scale for consistency
            center = np.mean(init_points, axis=0)
            centered = init_points - center
            norm = np.linalg.norm(centered)
            if norm > 1e-10:
                centered = centered / norm * 1.0
            else:
                centered = init_points
            
            score = -objective(centered)
            if score > best_score:
                best_score = score
                best_points = centered.copy()
                
    # Final sanity check
    if best_points is None:
        # Return the regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
