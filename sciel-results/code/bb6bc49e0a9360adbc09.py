"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    # Vectorized computation for speed
    # For each pair (i, j), compute area with all k > j
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    for i in range(n):
        for j in range(i + 1, n):
            v = points[j] - points[i]
            # Cross product with all remaining points
            # cross(v, p_k - p_i) = v_x * (p_k_y - p_i_y) - v_y * (p_k_x - p_i_x)
            diffs = points[j+1:] - points[i]
            crosses = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas) if len(areas) > 0 else float('inf')
            if local_min < min_area:
                min_area = local_min
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: if all points are collinear or fewer than 3, area is 0
        if len(points) < 3:
            return 0.0
        # Simple check for collinearity
        p0, p1, p2 = points[0], points[1], points[2]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) < 1e-10:
            return 0.0
        # Try again with a different subset or just return a small epsilon to avoid div by zero
        # But ConvexHull should handle it. If it fails, likely degenerate.
        return 1e-10

def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(params, time_limit, best_score_holder):
    """Objective function for minimization (negative score)."""
    if time.time() > time_limit:
        return 0.0  # Return a value that won't be selected as best if we stop, but need to be careful
    # params: 28 values (14 x 2)
    points = params.reshape((N, 2))
    # Normalize to unit area? No, the score is invariant to scaling.
    # But we can normalize to keep numbers stable.
    # Let's normalize by subtracting centroid and scaling to unit hull area?
    # Actually, the score is scale invariant. 
    # However, for numerical stability, let's normalize the points.
    centroid = np.mean(points, axis=0)
    points = points - centroid
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    points = points / np.sqrt(hull_area)  # Scale so hull area is 1
    s = _score(points)
    if s > best_score_holder[0]:
        best_score_holder[0] = s
        best_score_holder[1] = points.copy()
    return -s

def construct():
    start_time = time.time()
    time_limit = start_time + 140  # Leave 10s margin
    
    best_score_holder = [0.0, None]
    best_points = None
    
    # Strategy: Jittered regular 14-gon
    # Base points: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # We will optimize the radial jitter for each point + global affine transformation parameters?
    # The directive suggests: "optimize the global affine transformation (scale, rotation, shear) and the noise amplitude"
    # But since score is invariant to affine transformations (if we consider area ratios), 
    # wait, affine transformations preserve area ratios? 
    # Yes, any invertible affine transformation multiplies all areas by the same determinant factor.
    # So the score min_triangle_area / hull_area is invariant under any invertible affine transformation.
    # This means we can fix the affine transformation (e.g., identity) and just optimize the relative positions.
    # However, the "jittered" model implies deviations from the regular polygon.
    
    # Let's parameterize the points directly in 2D space, but start from the regular polygon.
    # We can use Differential Evolution to search the 28-dimensional space.
    # DE is global and robust.
    
    # Bounds for DE: points in [-2, 2] x [-2, 2]
    bounds = [(-2, 2)] * (2 * N)
    
    # Run Differential Evolution
    # To speed up, we might not be able to run DE with many iterations.
    # Let's try a few random restarts with local optimization (L-BFGS-B) on the soft-min surrogate?
    # The lesson says: "Soft-min surrogate ... enables escape from ring parameterization local optima"
    # But DE is a black-box global optimizer.
    
    # Let's try DE first.
    # maxiter needs to be small to fit in time.
    # Each eval of _objective takes ~0.1ms? 14 choose 3 = 364 triangles.
    # _min_triangle_area is O(N^3) but optimized with numpy. 
    # 364 triangles is small.
    
    # Let's estimate time per eval.
    # If 1ms per eval, we can do 1000 evals in 1s.
    # DE with popsize=15 (default is 15*N? No, 15 by default? Actually popsize=15 is default multiplier for N? No, popsize is the number of individuals. Default is 15*N? No, default is 15. Wait, scipy DE popsize default is 15. So 15 individuals.
    # If maxiter=100, that's 1500 evals.
    
    # Let's run DE with a time check.
    # We need to pass time_limit and best_score_holder.
    
    # To be safe, let's use a custom worker or just check time inside the objective.
    # DE doesn't support early stopping easily without a custom worker.
    # We can set maxiter low and check time after.
    
    # Alternative: Random restarts with L-BFGS-B on the smooth surrogate.
    # The lesson specifically mentions soft-min surrogate improving scores.
    # Let's implement the soft-min surrogate approach.
    
    def _soft_min_triangle_area(points, beta):
        """
        Compute soft-min of triangle areas using log-sum-exp.
        min(A_i) approx -1/beta * log(sum(exp(-beta * A_i)))
        """
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                v = points[j] - points[i]
                diffs = points[j+1:] - points[i]
                crosses = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
                tri_areas = 0.5 * np.abs(crosses)
                areas.extend(tri_areas)
        areas = np.array(areas)
        # Soft min
        min_area = np.min(areas)
        # Shift to avoid overflow
        shifted = -beta * (areas - min_area)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        soft_min = min_area - (1.0 / beta) * log_sum_exp
        return soft_min

    def _surrogate_objective(params, beta, time_limit, best_score_holder):
        if time.time() > time_limit:
            return 0.0
        points = params.reshape((N, 2))
        centroid = np.mean(points, axis=0)
        points = points - centroid
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        points = points / np.sqrt(hull_area)
        
        soft_min = _soft_min_triangle_area(points, beta)
        # Score approx: soft_min / 1.0 (since hull area is 1)
        # We want to maximize this, so minimize negative.
        s = soft_min
        if s > best_score_holder[0]:
            best_score_holder[0] = s
            best_score_holder[1] = points.copy()
        return -s

    # Run multiple restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    
    # Start from regular polygon
    initial_points = base_points.copy()
    
    # Try a few different initializations and betas
    # Beta 50 then 250 as per lesson
    
    for beta in [50, 250]:
        # Try 5 random restarts
        for restart in range(5):
            if time.time() > time_limit - 5:
                break
            
            if restart == 0:
                x0 = initial_points.flatten()
            else:
                # Jitter the initial points
                jitter = rng.normal(0, 0.1, size=(N, 2))
                x0 = (initial_points + jitter).flatten()
            
            # Normalize x0
            pts = x0.reshape((N, 2))
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            hull_area = _convex_hull_area(pts)
            if hull_area > 1e-10:
                pts = pts / np.sqrt(hull_area)
                x0 = pts.flatten()
            
            res = minimize(
                _surrogate_objective,
                x0,
                args=(beta, time_limit, best_score_holder),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 1000}
            )
            
            # Evaluate final point with exact score
            final_pts = res.x.reshape((N, 2))
            centroid = np.mean(final_pts, axis=0)
            final_pts = final_pts - centroid
            hull_area = _convex_hull_area(final_pts)
            if hull_area > 1e-10:
                final_pts = final_pts / np.sqrt(hull_area)
                s = _score(final_pts)
                if s > best_score_holder[0]:
                    best_score_holder[0] = s
                    best_score_holder[1] = final_pts.copy()
    
    # If we have a best point, return it. Otherwise, return the regular polygon.
    if best_score_holder[1] is not None:
        return best_score_holder[1]
    else:
        return base_points

# EVOLVE-BLOCK-END
