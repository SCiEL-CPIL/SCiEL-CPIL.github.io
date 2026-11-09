import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 600.0

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull."""
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _voronoi_centroids(points):
    """
    Compute the centroid of the Voronoi cell for each point.
    For points on the hull, the Voronoi cell is unbounded. 
    We approximate the centroid by clipping the unbounded rays to a large bounding box 
    or simply using the centroid of the bounded part if it exists, 
    but a more robust approach for Lloyd's in bounded domains is to project 
    or use a weighted average of neighbors. 
    However, standard Lloyd's for bounded domains often uses the centroid of the 
    intersection of the Voronoi cell with the domain boundary.
    
    For simplicity and robustness in this short time budget, we will compute the 
    centroid of the Voronoi vertices associated with each point. 
    If a point is on the hull, its Voronoi cell extends to infinity. 
    We handle this by only moving interior points via Voronoi centroids, 
    and optimizing hull points via a different mechanism (e.g., maintaining 
    a regular polygon or optimizing boundary energy).
    
    Actually, a better approach for a bounded convex region:
    1. Compute Voronoi diagram.
    2. For each point, collect the finite vertices of its Voronoi cell.
    3. If the cell is unbounded (hull point), we don't use Lloyd's directly on it 
       in the same way, or we clip to the convex hull of all points.
    
    Let's use a simpler proxy: 
    For interior points: move to centroid of Voronoi vertices.
    For hull points: we will keep them on a regular polygon or optimize them 
    separately to maintain convexity and spread.
    """
    n = len(points)
    centroids = points.copy()
    
    try:
        vor = Voronoi(points)
    except Exception:
        return points
        
    # Map each point to its Voronoi region
    # vor.regions[i] is a list of vertex indices. -1 means unbounded.
    for i in range(n):
        region = vor.regions[i]
        if len(region) == 0 or -1 in region:
            # Unbounded or empty: keep original position or handle specially
            # For Lloyd's in a box, we might project. 
            # But since we are optimizing hull separately, we leave hull points 
            # to be handled by the hull optimizer.
            # We will only update interior points here.
            # How to know if it's interior? 
            # A point is on the hull if it is part of the ConvexHull vertices.
            pass
            
    # Let's identify hull points explicitly
    try:
        hull = ConvexHull(points)
        hull_indices = set(hull.vertices)
    except Exception:
        hull_indices = set(range(n)) # Fallback: treat all as hull if hull fails
        
    for i in range(n):
        if i in hull_indices:
            continue # Skip hull points for Voronoi centroid update
            
        region = vor.regions[i]
        if len(region) > 0 and -1 not in region:
            verts = vor.vertices[region]
            if len(verts) > 0:
                centroids[i] = np.mean(verts, axis=0)
        # If bounded but empty (shouldn't happen for valid Voronoi), keep original
        
    return centroids

def _lloyd_relax(points, n_iters=20, step_size=1.0):
    """
    Perform Lloyd's algorithm iterations.
    Only moves interior points to their Voronoi centroids.
    Hull points are left unchanged in this step.
    """
    pts = points.copy()
    for _ in range(n_iters):
        if time.time() > _DEADLINE - 10:
            break
        new_pts = _voronoi_centroids(pts)
        # Only update interior points
        try:
            hull = ConvexHull(pts)
            hull_indices = set(hull.vertices)
        except Exception:
            hull_indices = set(range(len(pts)))
            
        for i in range(len(pts)):
            if i not in hull_indices:
                # Relaxation step: move towards centroid
                pts[i] = pts[i] * (1 - step_size) + new_pts[i] * step_size
                
        # Check convergence
        if np.linalg.norm(pts - points) < 1e-8:
            points = pts
            break
        points = pts
    return points

def _optimize_hull(pts, n_iters=50):
    """
    Optimize hull points to maximize the score, while keeping interior points 
    relatively fixed or re-relaxed.
    This is a simplified local search on the hull vertices.
    """
    n = len(pts)
    try:
        hull = ConvexHull(pts)
        hull_indices = np.array(hull.vertices)
    except Exception:
        return pts
        
    if len(hull_indices) < 3:
        return pts
        
    # We will perturb hull points slightly and see if score improves.
    # This is a simple hill-climbing on the hull.
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Sort hull points by angle to maintain order? 
    # ConvexHull vertices are in CCW order.
    
    for _ in range(n_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx in hull_indices:
            # Try small perturbations in radial direction
            center = np.mean(best_pts[hull_indices], axis=0)
            vec = best_pts[idx] - center
            norm = np.linalg.norm(vec)
            if norm < 1e-6: continue
            dir_vec = vec / norm
            
            # Try moving out and in
            for delta in [0.01, -0.01, 0.05, -0.05]:
                new_pts = best_pts.copy()
                new_pts[idx] = best_pts[idx] + delta * dir_vec
                
                # Ensure new point is still roughly in the right "spot" 
                # (don't cross other hull points). 
                # Simple check: recompute hull and ensure index is still on hull?
                # That's expensive. Let's just check score and convexity.
                if not np.all(np.isfinite(new_pts)):
                    continue
                
                # Check if new point set is valid (hull exists)
                try:
                    ConvexHull(new_pts)
                except Exception:
                    continue
                    
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_pts = new_pts
                    improved = True
                    break # Accept first improvement for this vertex
            
            # If improved, re-relax interior points? 
            # That might be too slow. Let's do a light relaxation.
            if improved:
                # Light Lloyd's on interior
                try:
                    hull = ConvexHull(best_pts)
                    hull_idx = set(hull.vertices)
                    for i in range(len(best_pts)):
                        if i not in hull_idx:
                             # Just one step of Lloyd's
                             # This is expensive if done for every step. 
                             # Let's skip full Lloyd's in this inner loop.
                             pass
                except Exception:
                    pass
                    
    return best_pts

def _generate_concentric_rings(n_total):
    """
    Generate initial points in concentric rings.
    Try different distributions:
    1. Center + Ring1 + Ring2 ...
    2. Ring1 + Ring2 ... (no center)
    """
    configs = []
    
    # Config 1: 1 center, then rings
    remaining = n_total - 1
    rings = []
    r = 1
    while remaining > 0:
        # Take up to 6 points per ring (hexagonal packing)
        take = min(remaining, 6)
        rings.append(take)
        remaining -= take
        r += 1
        
    if rings:
        configs.append((1, rings))
        
    # Config 2: No center, just rings starting from 3 or 4
    remaining = n_total
    rings = []
    # Start with a small inner ring
    inner_sizes = [3, 4, 5, 6]
    for inner in inner_sizes:
        if remaining >= inner:
            rem = remaining - inner
            rings = [inner]
            r = 1
            while rem > 0:
                take = min(rem, 6)
                rings.append(take)
                rem -= take
                r += 1
            configs.append((0, rings))
            
    return configs

def _build_points_from_rings(center_count, ring_counts, rng):
    """
    Build points based on ring structure.
    """
    pts = []
    cx, cy = 0.5, 0.5
    
    # Center points
    if center_count == 1:
        pts.append([cx, cy])
    elif center_count > 1:
        # Cluster around center
        for _ in range(center_count):
            r = rng.uniform(0, 0.05)
            theta = rng.uniform(0, 2*np.pi)
            pts.append([cx + r*np.cos(theta), cy + r*np.sin(theta)])
            
    radii = [0.2, 0.4, 0.6, 0.8, 1.0] # Max radius 1.0 -> area ~ pi
    # Actually, let's scale to fit in a unit square or circle.
    # Let's use a unit circle for max area.
    
    for i, count in enumerate(ring_counts):
        radius = radii[i] if i < len(radii) else 0.2 + 0.2*i
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        # Add some jitter
        angles += rng.uniform(-0.1, 0.1, count)
        for a in angles:
            x = cx + radius * np.cos(a)
            y = cy + radius * np.sin(a)
            pts.append([x, y])
            
    return np.array(pts)

def heilbronn_convex13() -> np.ndarray:
    n_total = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Get various structured initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    reg_13 = np.column_stack([0.5 + 0.45*np.cos(angles), 0.5 + 0.45*np.sin(angles)])
    inits.append(reg_13)
    
    # 2. Concentric Rings
    configs = _generate_concentric_rings(n_total)
    for center_count, ring_counts in configs:
        # Try a few random jitters for each config
        for _ in range(3):
            pts = _build_points_from_rings(center_count, ring_counts, rng)
            # Normalize to unit circle roughly
            # Find max distance from center
            center = np.mean(pts, axis=0)
            max_dist = np.max(np.linalg.norm(pts - center, axis=1))
            if max_dist > 0:
                pts = center + (pts - center) * (0.45 / max_dist)
            inits.append(pts)
            
    # 3. Random points (fallback)
    for _ in range(5):
        pts = rng.random((n_total, 2)) * 0.8 + 0.1
        inits.append(pts)

    # Optimize each initialization
    for init_pts in inits:
        if time.time() > _DEADLINE - 30:
            break
            
        current_pts = init_pts.copy()
        
        # Phase 1: Lloyd's Relaxation
        # This balances the density
        current_pts = _lloyd_relax(current_pts, n_iters=30, step_size=0.5)
        
        # Phase 2: Hull Optimization
        # Optimize the hull vertices to expand the hull area while maintaining min triangle area
        current_pts = _optimize_hull(current_pts, n_iters=20)
        
        # Phase 3: Final fine-tuning with Powell on all points
        # Use a soft-min objective for differentiability
        def obj_soft(x, temp=0.05):
            p = x.reshape(n_total, 2)
            ha = _hull_area(p)
            if ha <= 0: return 1e10
            # Soft min area
            idx = np.array(list(itertools.combinations(range(n_total), 3)))
            a = p[idx[:, 0]]
            b = p[idx[:, 1]]
            c = p[idx[:, 2]]
            areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
            neg_areas = -areas
            max_neg = np.max(neg_areas)
            log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
            soft_min = -temp * log_sum_exp
            return -soft_min / ha
            
        try:
            res = minimize(obj_soft, current_pts.flatten(), method='Powell',
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < 1e9:
                p_opt = res.x.reshape(n_total, 2)
                if np.all(np.isfinite(p_opt)):
                    s = _score(p_opt)
                    if s > best_score:
                        best_score = s
                        best_points = p_opt.copy()
        except Exception:
            pass
        
        # Also evaluate the pre-Powell state
        s = _score(current_pts)
        if s > best_score:
            best_score = s
            best_points = current_pts.copy()

    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
