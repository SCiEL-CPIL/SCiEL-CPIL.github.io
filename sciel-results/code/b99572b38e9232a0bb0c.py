"""
Heilbronn convex regions n=14: Multi-Start Nelder-Mead with Convexity Repair.
Strategy:
1. Use a fast, coarse Differential Evolution (DE) to generate a pool of diverse initial candidates.
2. For each candidate, perform local refinement using Nelder-Mead.
3. Implement an 'error_repair' mechanism: after each Nelder-Mead step, check for convexity violations.
   If violations occur (non-convex polygon), repair the points by projecting them onto the convex hull
   or re-sorting to enforce a valid convex configuration. This ensures the optimizer stays in the 
   valid search space and answers the question of whether repair improves child quality.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools
import sys

N = 14
_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]

def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _normalize_points(points):
    """Normalize points to unit area and centered at origin."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area < 1e-10:
            return points
        points = points / np.sqrt(area)
    except Exception:
        max_r = np.max(np.linalg.norm(points, axis=1))
        if max_r > 1e-10:
            points = points / max_r
    return points

def _is_convex(points):
    """
    Check if points form a strictly convex polygon in the given order.
    Assumes points are sorted by angle around centroid.
    Returns True if all cross products of consecutive edge vectors are positive.
    """
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = points[sorted_idx]
    
    n = len(sorted_pts)
    # Check cross products of consecutive edges
    for i in range(n):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i + 1) % n]
        p2 = sorted_pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-8: # Small tolerance for numerical stability
            return False
    return True

def _repair_convexity(points):
    """
    Error Repair: If the points are not convex, repair them.
    Strategy: 
    1. Compute convex hull.
    2. If number of hull vertices < N, some points are interior.
       We map the interior points to the boundary by projecting them onto the hull edges 
       or simply re-generating a regular polygon perturbed by the current shape's statistics.
       A simpler, robust repair: Project points onto the convex hull of the current set.
       However, this reduces the number of points. 
       
       Better Repair for this specific problem:
       Sort by angle. If not convex, the "dips" indicate non-convexity.
       We can enforce convexity by taking the convex hull and then redistributing the 
       points along the hull boundary to maintain N points.
       
       For simplicity and speed in the inner loop:
       If not convex, project each point onto the convex hull of the set.
       If a point is inside, it moves to the boundary.
       This changes the score but keeps the points in a "valid" convex configuration.
    """
    try:
        hull = ConvexHull(points)
        # If all points are on the hull, it's convex (assuming no collinear issues)
        if len(hull.vertices) == N:
            # Check strict convexity
            if _is_convex(points):
                return points
    except Exception:
        pass
    
    # Repair: Project onto convex hull
    # We need to map interior points to the boundary.
    # A simple heuristic: 
    # 1. Get hull vertices.
    # 2. For each point, if it's inside, project it onto the nearest hull edge.
    
    # This is computationally expensive if done naively for every step.
    # Alternative fast repair: 
    # If the polygon is non-convex, it likely has "reflex" angles.
    # We can smooth the points by averaging with a regular polygon?
    # Let's use a "soft" repair: 
    # Calculate the convex hull. If points are missing from the hull, 
    # we replace the interior points with points interpolated on the hull edges.
    
    # For the sake of the Nelder-Mead loop speed, we will use a simpler repair:
    # If not convex, we sort by angle and force radial monotonicity? 
    # No, let's just project onto the hull.
    
    # Efficient projection:
    # Use scipy.spatial.ConvexHull's equations? No, just check.
    
    # Actually, a very effective repair for this specific "min triangle area" problem
    # is to ensure points are in convex position. If they are not, the score is likely bad.
    # We can simply return the points of the convex hull, but we need N points.
    # So we take the hull vertices and fill the gaps with interpolated points.
    
    try:
        hull = ConvexHull(points)
        hull_verts = points[hull.vertices]
        # We have M <= N vertices. We need N points.
        # Distribute the remaining N-M points evenly along the hull edges.
        if len(hull_verts) == N:
            return points # Already convex (or close enough)
        
        # Interpolate
        new_points = []
        m = len(hull_verts)
        if m == 0:
             return points
        if m == 1:
             return np.tile(hull_verts[0], (N, 1))
        if m == 2:
             # Degenerate line
             for i in range(N):
                 t = i / (N - 1)
                 p = (1 - t) * hull_verts[0] + t * hull_verts[1]
                 new_points.append(p)
             return np.array(new_points)
        
        # m >= 3
        # We want N points on the perimeter.
        # Calculate edge lengths
        edge_lengths = []
        for i in range(m):
            p1 = hull_verts[i]
            p2 = hull_verts[(i + 1) % m]
            edge_lengths.append(np.linalg.norm(p2 - p1))
        total_perimeter = sum(edge_lengths)
        
        # We want to place N points evenly along the perimeter.
        # Step size in perimeter
        step = total_perimeter / N
        
        current_dist = 0.0
        new_points = []
        for i in range(N):
            target_dist = i * step
            
            # Find which edge this target_dist falls on
            cum_dist = 0.0
            for j in range(m):
                edge_len = edge_lengths[j]
                if cum_dist + edge_len >= target_dist:
                    # Target is on this edge
                    p1 = hull_verts[j]
                    p2 = hull_verts[(j + 1) % m]
                    # Distance from p1 to target
                    dist_from_p1 = target_dist - cum_dist
                    t = dist_from_p1 / edge_len if edge_len > 0 else 0
                    p = (1 - t) * p1 + t * p2
                    new_points.append(p)
                    break
                cum_dist += edge_len
            
            if len(new_points) < N:
                 # Fallback if loop didn't break (should not happen)
                 new_points.append(hull_verts[0])
        
        return np.array(new_points)
    
    except Exception:
        # If repair fails, return original points (optimizer will penalize via score)
        return points

def _objective_with_repair(flat_points):
    """
    Objective function for Nelder-Mead.
    1. Reshape points.
    2. Normalize.
    3. Repair if not convex.
    4. Calculate score.
    5. Return negative score.
    """
    points = flat_points.reshape(N, 2)
    points = _normalize_points(points)
    
    # Error Repair Step
    # Only repair if we suspect non-convexity to save time?
    # _is_convex is O(N), which is fast.
    if not _is_convex(points):
        points = _repair_convexity(points)
        points = _normalize_points(points)
        
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Generate diverse initial candidates
    # Use a fast DE to get a few good starting points
    init_candidates = []
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_candidates.append(reg_points)
    
    # Slight perturbations of regular polygon
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (N, 2))
        pert_points = reg_points + noise
        init_candidates.append(pert_points)
        
    # Fast DE for more diversity
    try:
        bounds = [(-5, 5)] * (2 * N)
        # Very short DE run just to get diverse points
        de_result = differential_evolution(
            _objective_with_repair,
            bounds,
            seed=42,
            maxiter=20,
            popsize=10,
            tol=1e-4,
            polish=False,
            init='sobol'
        )
        # Extract some points from the DE history? 
        # DE doesn't easily expose intermediate populations without hooks.
        # Instead, we'll just use the result and a few random points.
        de_points = de_result.x.reshape(N, 2)
        init_candidates.append(de_points)
        
        # Add some random points
        for _ in range(3):
            rand_points = np.random.uniform(-2, 2, (N, 2))
            init_candidates.append(rand_points)
    except Exception:
        pass

    # Sort candidates by score (descending)
    scored_candidates = []
    for pts in init_candidates:
        pts = _normalize_points(pts)
        if not _is_convex(pts):
            pts = _repair_convexity(pts)
            pts = _normalize_points(pts)
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_candidates[0][0]
    best_points = scored_candidates[0][1]
    
    # 2. Multi-Start Nelder-Mead Refinement
    # We have time budget. Let's allocate time to each start.
    # Say, 5 seconds per start. 
    # If we have 10 starts, that's 50 seconds. 
    # We can do more if we are fast.
    
    # Limit to top 10 candidates to ensure time
    top_candidates = scored_candidates[:10]
    
    for i, (s_init, pts_init) in enumerate(top_candidates):
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                _objective_with_repair,
                pts_init.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            cur_points = res.x.reshape(N, 2)
            cur_points = _normalize_points(cur_points)
            
            # Final repair
            if not _is_convex(cur_points):
                cur_points = _repair_convexity(cur_points)
                cur_points = _normalize_points(cur_points)
                
            cur_score = _score(cur_points)
            
            if cur_score > best_score:
                best_score = cur_score
                best_points = cur_points
                
        except Exception:
            continue
            
    # 3. Final Polish with L-BFGS-B if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_with_repair,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cur_points = res.x.reshape(N, 2)
            cur_points = _normalize_points(cur_points)
            if not _is_convex(cur_points):
                cur_points = _repair_convexity(cur_points)
                cur_points = _normalize_points(cur_points)
            cur_score = _score(cur_points)
            if cur_score > best_score:
                best_score = cur_score
                best_points = cur_points
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
