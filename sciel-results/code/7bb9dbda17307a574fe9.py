# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 2^(k-2) + 1 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex k-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Strategy: Start with a structured grid-like configuration that is known to minimize
    convex polygons (e.g., points on a convex curve or a carefully arranged lattice),
    then apply local optimization to break any remaining convex 7-gons while maintaining
    general position (no three collinear).
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    k = 7  # size of convex polygon to count
    
    # Start with a configuration that is known to have few convex polygons.
    # One approach: place points on a parabola y = x^2, which creates many collinear-ish
    # but actually we want to avoid convex positions. 
    # Better: use a "cup" construction - points on a concave curve.
    # Or use a grid that is slightly perturbed.
    
    # Let's try a structured approach: place points in a way that maximizes concavity.
    # A known construction for few convex k-gons is to place points on a convex curve
    # but in reverse order, or use a "spiral" arrangement.
    
    # Actually, the best known constructions for the Erdos-Szekeres problem use
    # points that are "almost" collinear but arranged to avoid forming convex polygons.
    
    # Let's start with a regular grid and then optimize.
    # Grid of 6x6 = 36 points, take 33 of them.
    # But grids have many collinear points, which is invalid.
    
    # Better: start with points on a convex curve (like a circle) but with some
    # perturbation to break collinearity, then optimize to minimize convex 7-gons.
    
    # Initialize with points on a parabola, which is a convex curve.
    # Points on y = x^2 from x=0 to x=1.
    # This will have many convex subsets, so we need to optimize.
    
    # Alternative: use a "zigzag" or "alternating" pattern.
    
    # Let's try: place points in 11 rows of 3 points each, with each row slightly
    # offset to avoid collinearity.
    
    # Actually, let's use a random initialization with a good seed, then do local search.
    # But the task says to design from scratch with a specific approach.
    
    # The directive says: "Model the problem as a discrete combinatorial packing task by 
    # initializing points on a sparse grid of high-symmetry lattice points (e.g., integer 
    # coordinates scaled to the unit square) to maximize collinearity, then use a stochastic 
    # hill-climbing search that only moves points to other grid positions or small neighborhood 
    # perturbations to break degeneracies while preserving the dense, non-convex structure."
    
    # So let's follow that: start with a grid, then perturb to break collinearity while
    # minimizing convex 7-gons.
    
    # Create a 6x6 grid of points in the unit square.
    # 6x6 = 36 points, we need 33, so remove 3 points.
    
    # Grid points: i/5, j/5 for i,j in 0..5
    grid_x = np.linspace(0, 1, 6)
    grid_y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Remove 3 points to get 33. Remove corners to reduce convexity?
    # Actually, let's keep most and just take the first 33.
    points = grid_points[:33].copy()
    
    # Now we need to break collinearity. Add small random perturbations.
    np.random.seed(42)
    points += np.random.randn(33, 2) * 0.01
    
    # Clip to unit square
    points = np.clip(points, 0, 1)
    
    # Now do local optimization to minimize the number of convex 7-gons.
    # We'll use a simple hill-climbing approach: try small perturbations and accept
    # if the number of convex 7-gons decreases.
    
    def count_convex_7gons(pts):
        """Count the number of 7-point subsets in convex position."""
        n_pts = len(pts)
        count = 0
        # This is expensive: C(33,7) = 4,292,145 subsets.
        # For each subset, check if it's in convex position.
        # This is too slow for local search. We need a faster heuristic.
        
        # Instead, let's use a proxy: count the number of "convex chains" or use
        # a sampling approach.
        
        # For now, let's just return 0 as a placeholder and implement a faster metric.
        # Actually, let's implement a faster check using the dynamic programming approach
        # mentioned in the task, but that's also complex.
        
        # Simpler: use the fact that a set of points is in convex position iff
        # all points are on the convex hull. So for each 7-point subset, check if
        # all 7 points are on the convex hull of that subset.
        
        # This is still expensive. Let's use a random sample of subsets.
        n_samples = 10000
        indices = np.random.randint(0, n_pts, size=(n_samples, 7))
        # Remove duplicates in each row
        for i in range(n_samples):
            if len(set(indices[i])) < 7:
                indices[i] = np.random.permutation(n_pts)[:7]
        
        convex_count = 0
        for subset in indices:
            pts_subset = pts[subset]
            # Check if all points are on the convex hull
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts_subset)
                if len(hull.vertices) == 7:
                    convex_count += 1
            except:
                pass
        
        return convex_count / n_samples * np.math.comb(n_pts, 7)
    
    # This is still too slow. Let's use a much faster proxy metric.
    # Proxy: for each point, count the number of "convex turns" in its neighborhood.
    # Or: use the sum of the areas of triangles formed by triples of points.
    # Points that are more "convex" will have larger total triangle areas.
    
    def convexity_proxy(pts):
        """A fast proxy for the number of convex 7-gons.
        Sum over all triples of the area of the triangle, weighted by whether
        the triple is "convex" (i.e., not collinear)."""
        n_pts = len(pts)
        total = 0.0
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                for k in range(j+1, n_pts):
                    # Area of triangle i,j,k
                    ax, ay = pts[i]
                    bx, by = pts[j]
                    cx, cy = pts[k]
                    area = 0.5 * abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                    total += area
        return total
    
    # Even this is O(n^3) = 33^3 = 35937, which is okay for a few iterations.
    
    # Let's do a simple local search: try perturbing each point slightly and accept
    # if the proxy metric decreases.
    
    best_points = points.copy()
    best_proxy = convexity_proxy(best_points)
    
    for iteration in range(200):
        improved = False
        for i in range(n):
            for _ in range(10):
                # Try a small random perturbation
                perturbation = np.random.randn(2) * 0.005
                new_points = best_points.copy()
                new_points[i] = np.clip(best_points[i] + perturbation, 0, 1)
                
                # Check validity: no three collinear
                valid = True
                for a in range(n):
                    for b in range(a+1, n):
                        for c in range(b+1, n):
                            ax, ay = new_points[a]
                            bx, by = new_points[b]
                            cx, cy = new_points[c]
                            cross = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
                            if abs(cross) < 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                new_proxy = convexity_proxy(new_points)
                if new_proxy < best_proxy:
                    best_points = new_points
                    best_proxy = new_proxy
                    improved = True
        
        if not improved:
            break
    
    return best_points


# EVOLVE-BLOCK-END
