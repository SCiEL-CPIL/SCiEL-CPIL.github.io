# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon (vertices in CCW order) that maximizes the minimum per-vertex
    equidistance score.

    Design:
      * Uses 4-fold rotational symmetry: the 20 vertices are defined by 5 distinct
        angular offsets and 5 radial offsets in the first quadrant, repeated 4 times
        with 90-degree rotations. This reduces the search space from 20 dimensions to 10.
      * Guarantees convexity by checking cross products of consecutive edges.
      * Uses Differential Evolution for global search and Nelder-Mead for local refinement.

    Returns:
        np.ndarray of shape (20, 2), finite, in cyclic (CCW) order.
    """
    n = 20
    start = time.time()
    deadline = 540.0

    # ---------- scoring helpers ----------
    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        raw = np.mean(np.maximum(sel / D, D / sel))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _polygon_score(pts):
        p = _normalize(pts)
        if p is None:
            return 0.0
        m = p.shape[0]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        best = 1.0
        for i in range(m):
            other = np.delete(dist[i], i)
            other.sort()
            s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _is_convex(pts):
        """Check if the polygon defined by pts is strictly convex."""
        n_pts = len(pts)
        if n_pts < 3:
            return False
        prev_sign = 0
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i + 1) % n_pts]
            p2 = pts[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-10:
                return False
            sign = 1 if cross > 0 else -1
            if prev_sign == 0:
                prev_sign = sign
            elif sign != prev_sign:
                return False
        return True

    def _build_polygon_from_params(params):
        """
        Takes 10 parameters: 5 angles in [0, pi/2) and 5 radii in [0.5, 1.5].
        Constructs the 20-point polygon with 4-fold symmetry.
        """
        angles = np.sort(params[0:5])
        radii = params[5:10]
        
        # Ensure angles are distinct and within bounds
        if np.any(angles < 0) or np.any(angles >= np.pi/2):
            return None
        if np.any(np.diff(angles) < 1e-8):
            return None
        # Check wrap-around gap in the quadrant
        wrap_gap = (angles[0] + np.pi/2) - angles[4]
        if wrap_gap < 1e-8:
            return None
            
        # Ensure radii are positive
        if np.any(radii <= 0):
            return None

        # Construct full angles and radii
        # Q1: angles, radii
        # Q2: angles + pi/2, radii
        # Q3: angles + pi, radii
        # Q4: angles + 3*pi/2, radii
        
        q1_angles = angles
        q2_angles = angles + np.pi/2
        q3_angles = angles + np.pi
        q4_angles = angles + 3*np.pi/2
        
        all_angles = np.concatenate([q1_angles, q2_angles, q3_angles, q4_angles])
        all_radii = np.tile(radii, 4)
        
        # Sort by angle to get cyclic order
        sort_idx = np.argsort(all_angles)
        final_angles = all_angles[sort_idx]
        final_radii = all_radii[sort_idx]
        
        x = final_radii * np.cos(final_angles)
        y = final_radii * np.sin(final_angles)
        pts = np.stack([x, y], axis=1)
        
        # Check convexity
        if not _is_convex(pts):
            return None
            
        return pts

    def _obj_symmetric(params):
        pts = _build_polygon_from_params(params)
        if pts is None:
            return 1.0 # Bad score (minimizing negative score, so large positive is bad)
        return -_polygon_score(pts)

    # ---------- baseline: regular polygon ----------
    # Regular polygon: angles 0, pi/10, 2pi/10, 3pi/10, 4pi/10. Radii all 1.
    base_angles = np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10])
    base_radii = np.ones(5)
    base_params = np.concatenate([base_angles, base_radii])
    
    best_pts = _build_polygon_from_params(base_params)
    if best_pts is None:
        # Fallback
        base_angles_full = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([np.cos(base_angles_full), np.sin(base_angles_full)], axis=1)
        
    best_score = _polygon_score(best_pts)
    best_params = base_params.copy()

    # ---------- Global Search: Differential Evolution ----------
    # Bounds: 5 angles in [0, pi/2], 5 radii in [0.5, 1.5]
    bounds = [(0.0, np.pi/2) for _ in range(5)] + [(0.5, 1.5) for _ in range(5)]
    
    try:
        res = differential_evolution(
            _obj_symmetric,
            bounds,
            seed=20240611,
            maxiter=2000,
            popsize=30,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 120.0 else False
            ),
        )
        if res is not None and res.x is not None:
            pts = _build_polygon_from_params(res.x)
            if pts is not None:
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
                    best_params = res.x.copy()
    except Exception:
        pass

    # ---------- Local Refinement: Nelder-Mead ----------
    if (time.time() - start) < deadline - 30.0:
        try:
            x0 = best_params.copy()
            
            r2 = minimize(
                _obj_symmetric, 
                x0, 
                method="Nelder-Mead",
                options={
                    "maxiter": 10000,
                    "xatol": 1e-10,
                    "fatol": 1e-12,
                    "adaptive": True
                }
            )
            
            if r2 is not None and r2.x is not None:
                pts = _build_polygon_from_params(r2.x)
                if pts is not None:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
        except Exception:
            pass

    # ---------- Final Safety ----------
    if _polygon_score(best_pts) <= 0.0:
        # Fallback to regular polygon
        base_angles_full = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([np.cos(base_angles_full), np.sin(base_angles_full)], axis=1)
        
    return best_pts

# EVOLVE-BLOCK-END
