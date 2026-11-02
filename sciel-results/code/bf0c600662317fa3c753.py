import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Pre-compute all triangle index combinations
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)  # shape (364, 3)
    
    # Fixed angles for the 14 points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    sin_delta = np.sin(2 * np.pi / N)

    # Number of harmonics to use for the radial expansion
    M = 5
    
    def get_radii_from_coeffs(coeffs):
        """
        coeffs: array of length 2*M containing [a1, b1, a2, b2, ..., aM, bM]
        Returns: array of N radii
        """
        r = np.ones(N)
        for m in range(1, M + 1):
            a = coeffs[2 * (m - 1)]
            b = coeffs[2 * (m - 1) + 1]
            r += a * np.cos(m * angles) + b * np.sin(m * angles)
        r = np.maximum(r, 0.1)
        return r

    def compute_score_from_pts(pts):
        """
        Compute min triangle area / convex hull area given Cartesian points.
        """
        # Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, .volume is area
        except Exception:
            # Fallback: use polygon area formula assuming CCW ordering
            # Shoelace formula
            x = pts[:, 0]
            y = pts[:, 1]
            hull_area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        
        if hull_area < 1e-10:
            return 0.0

        # Vectorized min triangle area computation
        pi = pts[tri_indices[:, 0]]  # shape (364, 2)
        pj = pts[tri_indices[:, 1]]  # shape (364, 2)
        pk = pts[tri_indices[:, 2]]  # shape (364, 2)
        
        v1 = pj - pi  # shape (364, 2)
        v2 = pk - pi  # shape (364, 2)
        
        # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        score = min_area / hull_area
        return score

    def compute_score_from_radii(r):
        """
        Compute min triangle area / convex hull area given radii.
        """
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts = np.column_stack([x, y])
        return compute_score_from_pts(pts)

    def neg_score_coeffs(coeffs):
        r = get_radii_from_coeffs(coeffs)
        score = compute_score_from_radii(r)
        return -score

    def neg_score_radii(r_flat):
        r = np.maximum(r_flat, 0.1)
        score = compute_score_from_radii(r)
        return -score

    def neg_score_cartesian(pts_flat):
        """
        Objective for direct Cartesian coordinate optimization.
        pts_flat: array of length 2*N containing [x0, y0, x1, y1, ..., xN-1, yN-1]
        """
        pts = pts_flat.reshape(N, 2)
        score = compute_score_from_pts(pts)
        return -score

    # Initial guess: Regular 14-gon
    best_r = np.ones(N)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score_val = compute_score_from_pts(best_pts)
    
    # Phase 1: Optimize Fourier coefficients (low dimensional, smooth manifold)
    try:
        x0_coeffs = np.zeros(2 * M)
        res1 = minimize(neg_score_coeffs, x0_coeffs, method='Powell',
                        options={'maxiter': 800, 'ftol': 1e-10, 'disp': False})
        
        if res1.fun < 0:
            score_val = -res1.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_r = get_radii_from_coeffs(res1.x)
                best_pts = np.column_stack([best_r * np.cos(angles), best_r * np.sin(angles)])
    except Exception:
        pass

    # Phase 2: Refine specific radii
    if time.time() < deadline - 20:
        try:
            r_start = best_r.copy()
            res2 = minimize(neg_score_radii, r_start, method='Powell',
                            options={'maxiter': 400, 'ftol': 1e-12, 'disp': False})
            
            if res2.fun < 0:
                score_val = -res2.fun
                r_candidate = np.maximum(res2.x, 0.1)
                score_check = compute_score_from_radii(r_candidate)
                if score_check > best_score_val:
                    best_score_val = score_check
                    best_r = r_candidate
                    best_pts = np.column_stack([best_r * np.cos(angles), best_r * np.sin(angles)])
        except Exception:
            pass

    # Phase 3: Direct Cartesian optimization - dual lattice approach
    # This allows non-radial, non-Fourier shape deformations
    if time.time() < deadline - 15:
        try:
            # Start from current best in Cartesian
            pts_flat = best_pts.flatten()
            
            # Use Powell for the full 28-dimensional space
            res3 = minimize(neg_score_cartesian, pts_flat, method='Powell',
                           options={'maxiter': 600, 'ftol': 1e-12, 'disp': False})
            
            if res3.fun < 0:
                score_val = -res3.fun
                pts_candidate = res3.x.reshape(N, 2)
                score_check = compute_score_from_pts(pts_candidate)
                if score_check > best_score_val:
                    best_score_val = score_check
                    best_pts = pts_candidate
        except Exception:
            pass

    # Phase 4: Nelder-Mead polish on Cartesian coordinates
    if time.time() < deadline - 5:
        try:
            pts_flat = best_pts.flatten()
            res4 = minimize(neg_score_cartesian, pts_flat, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            if res4.fun < 0:
                score_val = -res4.fun
                pts_candidate = res4.x.reshape(N, 2)
                score_check = compute_score_from_pts(pts_candidate)
                if score_check > best_score_val:
                    best_score_val = score_check
                    best_pts = pts_candidate
        except Exception:
            pass

    # Phase 5: Multi-start refinement with perturbed initial points
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            try:
                # Add small random perturbation
                noise_scale = 0.05 * (1 - trial * 0.3)
                pts_perturbed = best_pts + rng.normal(0, noise_scale, (N, 2))
                
                res5 = minimize(neg_score_cartesian, pts_perturbed.flatten(), method='Powell',
                               options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
                
                if res5.fun < 0:
                    score_val = -res5.fun
                    pts_candidate = res5.x.reshape(N, 2)
                    score_check = compute_score_from_pts(pts_candidate)
                    if score_check > best_score_val:
                        best_score_val = score_check
                        best_pts = pts_candidate
            except Exception:
                pass

    # Final validation: ensure points form a valid convex position
    # Check that all points are vertices of the convex hull
    try:
        hull = ConvexHull(best_pts)
        if len(hull.vertices) < N:
            # Some points are interior; fall back to best radial result
            x = best_r * np.cos(angles)
            y = best_r * np.sin(angles)
            best_pts = np.column_stack([x, y])
    except Exception:
        pass

    return best_pts
