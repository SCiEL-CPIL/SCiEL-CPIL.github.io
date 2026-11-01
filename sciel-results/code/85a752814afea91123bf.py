"""Heilbronn convex regions n=14: maximize minimum triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    # Use vectorized computation for speed
    # For each pair (i,j), compute cross product with all other points
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            # Cross product with all points k != i, j
            # cross(v, pk - pi) = v[0]*(pk[1]-pi[1]) - v[1]*(pk[0]-pi[0])
            diffs = pts - pts[i]
            cross_vals = v[0] * diffs[:, 1] - v[1] * diffs[:, 0]
            # Exclude i and j themselves
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross_vals[mask])
            min_area = min(min_area, np.min(areas))
    return min_area


def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points collinear or degenerate
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return -1.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _objective(x):
    """Objective to minimize (negative score)."""
    pts = np.array(x).reshape(-1, 2)
    s = _score(pts)
    if s < 0:
        return 1.0  # Large penalty for invalid
    return -s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy: 
    # 1. Start with regular polygon (good initial guess)
    # 2. Use differential evolution for global search
    # 3. Refine with L-BFGS-B local optimization
    
    # Regular polygon as initial guess
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(pts_reg)
    best_pts = pts_reg.copy()
    
    # Fix p0 at origin, p1 at (1,0) to remove translation and rotation invariance
    # This reduces variables from 2*N to 2*(N-2) = 24
    def decode(x):
        """Decode 24-d vector to 14 points with p0=(0,0), p1=(1,0)."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def encode(pts):
        """Encode 14 points to 24-d vector (drop p0, p1)."""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
    def obj_reduced(x):
        """Objective on reduced parameterization."""
        pts = decode(x)
        s = _score(pts)
        if s < 0:
            return 1.0
        return -s
    
    # Define bounds for the 24 free coordinates
    # Points should be in a reasonable range; the convex hull area normalization
    # makes this somewhat scale-invariant, but we need bounds for the optimizer
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60.0:
        try:
            result_de = differential_evolution(
                obj_reduced,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                init='sobol',
                callback=None,
                updating='deferred'
            )
            if result_de.success or result_de.fun < -best_score:
                x_opt = result_de.x
                pts_opt = decode(x_opt)
                s_opt = _score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_pts = pts_opt
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B
    if time.time() < deadline - 20.0:
        try:
            # Start from best found so far
            x_start = encode(best_pts)
            result_lbfgs = minimize(
                obj_reduced,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_lbfgs.success or result_lbfgs.fun < -best_score:
                x_opt = result_lbfgs.x
                pts_opt = decode(x_opt)
                s_opt = _score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_pts = pts_opt
        except Exception:
            pass
    
    # Phase 3: Try multiple random restarts with local optimization
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            try:
                # Random initial point
                x_rand = rng.uniform(-1.5, 1.5, size=2 * (N - 2))
                result_r = minimize(
                    obj_reduced,
                    x_rand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = result_r.x
                pts_opt = decode(x_opt)
                s_opt = _score(pts_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_pts = pts_opt
            except Exception:
                continue
    
    # Phase 4: One more final polish from the best
    if time.time() < deadline - 5.0:
        try:
            x_start = encode(best_pts)
            result_final = minimize(
                obj_reduced,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = result_final.x
            pts_opt = decode(x_opt)
            s_opt = _score(pts_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_pts = pts_opt
        except Exception:
            pass
    
    return best_pts
