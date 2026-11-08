import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    
    num_tris = len(tri_indices)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        
        if min_area < 0:
            return 0.0
        return min_area / area_hull

    def compute_smooth_score(points, beta):
        """
        Compute -softmin(areas) / hull_area.
        This is a smooth surrogate for the exact score.
        """
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 1e10 # Large penalty
            
        if area_hull < 1e-9:
            return 1e10

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min approximation: -1/beta * log(sum(exp(-beta * areas)))
        # To avoid overflow, we subtract the min area
        min_a = np.min(areas)
        exp_terms = np.exp(-beta * (areas - min_a))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = min_a - log_sum / beta
        
        if soft_min_val < 0:
            return 1e10
            
        return -soft_min_val / area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def polish_smooth(pts, beta=100.0):
        """
        Use L-BFGS-B to minimize the smooth surrogate.
        """
        def objective(x_flat):
            pts_test = x_flat.reshape(N, 2)
            return compute_smooth_score(pts_test, beta)
        
        x0 = pts.flatten()
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            return res.x.reshape(N, 2)
        except Exception:
            return pts

    def polish_exact(pts, max_iter=200):
        """
        Use Nelder-Mead or Powell to polish the exact (nonsmooth) score.
        """
        def objective(x_flat):
            pts_test = x_flat.reshape(N, 2)
            if time.time() > start_time + time_limit - 5:
                return 1e10
            s = compute_score(pts_test)
            return -s
        
        x0 = pts.flatten()
        try:
            # Powell is often more robust for this type of problem
            res = opt.minimize(objective, x0, method='Powell', 
                               options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6})
            return res.x.reshape(N, 2)
        except Exception:
            return pts

    # Generate diverse initial candidates
    rng = np.random.RandomState(42)
    
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    candidates.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Two concentric rings
    for n1 in [4, 5, 6, 7, 8]:
        n2 = N - n1
        if n2 < 3:
            continue
        for r1 in [0.3, 0.5, 0.7, 0.85]:
            outer = _regular_polygon(n1, 1.0)
            inner = _regular_polygon(n2, r1)
            angle_shift = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(angle_shift), -np.sin(angle_shift)],
                           [np.sin(angle_shift), np.cos(angle_shift)]])
            inner = inner @ rot.T
            candidates.append(np.vstack([outer, inner]))
    
    # 4. Random points in unit circle
    for _ in range(20):
        r = rng.uniform(0, 1, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # 5. Random perturbations of regular polygon
    for _ in range(20):
        base = _regular_polygon(N, 1.0)
        pts = base + rng.randn(N, 2) * 0.2
        candidates.append(pts)
        
    # 6. Grid-like structures
    # e.g. points on a 4x4 grid subset
    # We can just add a few more random ones
    
    # Phase 1: Quick evaluation of all candidates
    scored_candidates = []
    for pts in candidates:
        if time.time() > start_time + time_limit - 10:
            break
        score = compute_score(pts)
        scored_candidates.append((score, pts))
    
    # Sort by score and keep top K for polishing
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:10]
    
    best_score = top_candidates[0][0] if top_candidates else 0.0
    best_points = top_candidates[0][1] if top_candidates else candidates[0]

    # Phase 2: Polish top candidates
    # First, use smooth L-BFGS-B to get close to a good local optimum
    # Then, use Powell/NM to polish the exact objective
    
    for score, pts in top_candidates:
        if time.time() > start_time + time_limit - 15:
            break
        
        # Step 2a: Smooth polish
        smooth_pts = polish_smooth(pts, beta=100.0)
        
        # Step 2b: Exact polish
        polished_pts = polish_exact(smooth_pts, max_iter=100)
        
        polished_score = compute_score(polished_pts)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts

    # Phase 3: If time remains, try a few more random restarts with the full pipeline
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 20:
        # Estimate time per full pipeline (smooth + exact)
        # L-BFGS-B might take 1-2s, Powell 1-2s. Total ~3-4s per run.
        runs_possible = int(remaining_time / 5)
        
        for _ in range(runs_possible):
            if time.time() > start_time + time_limit - 5:
                break
            
            # Generate a new random candidate near the best found
            # Or just a random one
            r = rng.uniform(0, 1, N)
            theta = rng.uniform(0, 2 * np.pi, N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Or perturb best
            if rng.rand() > 0.5:
                pts = best_points + rng.randn(N, 2) * 0.1
            
            smooth_pts = polish_smooth(pts, beta=100.0)
            polished_pts = polish_exact(smooth_pts, max_iter=50)
            polished_score = compute_score(polished_pts)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts

    return best_points
