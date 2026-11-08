import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
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
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _stochastic_refine(points, max_time=10.0):
        """
        Fast stochastic hill climbing.
        Evaluates multiple perturbations in parallel using NumPy.
        """
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        start_refine = time.time()
        
        # Adaptive step size
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 50
        rng = np.random.RandomState(42)
        
        while time.time() - start_refine < max_time:
            if time.time() > start_time + time_limit - 2:
                break
                
            # Generate batch of perturbations
            n_samples = 100
            # Random directions
            dirs = rng.randn(n_samples, N, 2)
            dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
            
            # Apply perturbations
            candidates = pts[None, :, :] + dirs * step_size
            
            # Normalize all candidates (vectorized)
            # Note: ConvexHull is not easily vectorized, so we loop, but it's fast enough for 100
            # Alternatively, we can skip normalization in the loop if we assume small perturbations
            # but normalization is crucial for the score.
            # Let's do a quick vectorized normalization approximation or loop.
            # Looping 100 ConvexHulls is ~5-10ms. Acceptable.
            
            scores = []
            valid_indices = []
            for i in range(n_samples):
                c = candidates[i]
                c = _normalize_points(c)
                s = compute_score(c)
                scores.append(s)
                if s > best_score:
                    valid_indices.append(i)
            
            scores = np.array(scores)
            max_score = np.max(scores)
            
            if max_score > best_score:
                best_idx = np.argmax(scores)
                pts = _normalize_points(candidates[best_idx])
                best_score = max_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    step_size *= 0.8
                    no_improve_count = 0
                    if step_size < 1e-4:
                        break

        return pts, best_score

    def _differential_evolution():
        """
        Global search using Differential Evolution.
        """
        # Bounds: [-2, 2] for x and y
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        # Objective function for DE
        def de_objective(x_flat):
            if time.time() > start_time + time_limit - 5:
                return 0.0
            pts = x_flat.reshape(N, 2)
            pts = _normalize_points(pts)
            return -compute_score(pts) # Minimize negative score

        # Run DE
        # Popsize 15, maxiter 200. 
        # DE is robust but slow. We limit it.
        try:
            res = opt.differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50, # Keep low to ensure we have time for refinement
                popsize=15,
                tol=1e-5,
                polish=False, # We will do our own refinement
                init='sobol'
            )
            best_x = res.x
            best_score = -res.fun
            return best_x.reshape(N, 2), best_score
        except Exception as e:
            # Fallback to regular polygon if DE fails
            pts = _regular_polygon(N, 1.0)
            return pts, compute_score(pts)

    # Strategy:
    # 1. Run DE to find a good global candidate.
    # 2. Run stochastic refinement on the DE result.
    # 3. Also run stochastic refinement on a few structured initializations (regular polygon, etc.) in parallel-ish (sequentially but quickly).
    # 4. Return the best.

    best_points = None
    best_score = -float('inf')

    # Candidate 1: Differential Evolution
    if time.time() < start_time + time_limit - 30:
        de_pts, de_score = _differential_evolution()
        if de_score > best_score:
            best_score = de_score
            best_points = de_pts.copy()
    
    # Candidate 2: Stochastic Refinement from Regular Polygon
    if time.time() < start_time + time_limit - 10:
        reg_pts = _regular_polygon(N, 1.0)
        reg_pts, reg_score = _stochastic_refine(reg_pts, max_time=15.0)
        if reg_score > best_score:
            best_score = reg_score
            best_points = reg_pts.copy()

    # Candidate 3: Stochastic Refinement from "Two Rings" (7 outer, 6 inner)
    if time.time() < start_time + time_limit - 10:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, 0.5, offset=np.pi/6)
        init_pts = np.vstack([outer, inner])
        ring_pts, ring_score = _stochastic_refine(init_pts, max_time=15.0)
        if ring_score > best_score:
            best_score = ring_score
            best_points = ring_pts.copy()

    # Candidate 4: Stochastic Refinement from "12 Outer, 1 Center"
    if time.time() < start_time + time_limit - 10:
        outer = _regular_polygon(12, 1.0)
        center = np.array([[0.0, 0.0]])
        init_pts = np.vstack([outer, center])
        center_pts, center_score = _stochastic_refine(init_pts, max_time=15.0)
        if center_score > best_score:
            best_score = center_score
            best_points = center_pts.copy()

    # Final Polish: If we have time, do a very fine stochastic refinement on the best found
    if best_points is not None and time.time() < start_time + time_limit - 2:
        best_points, best_score = _stochastic_refine(best_points, max_time=5.0)

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
