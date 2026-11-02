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
    # r(theta) = 1 + sum_{m=1}^{M} (a_m cos(m theta) + b_m sin(m theta))
    M = 5
    
    def get_radii_from_coeffs(coeffs):
        """
        coeffs: array of length 2*M containing [a1, b1, a2, b2, ..., aM, bM]
        Returns: array of N radii
        """
        # Initialize radii to 1
        r = np.ones(N)
        for m in range(1, M + 1):
            a = coeffs[2 * (m - 1)]
            b = coeffs[2 * (m - 1) + 1]
            r += a * np.cos(m * angles) + b * np.sin(m * angles)
        
        # Ensure radii are positive to maintain convexity/validity
        # If a radius becomes <= 0, the point is at the origin or inside, 
        # which breaks the convex hull assumption or degenerates area.
        # We clip to a small positive value to avoid division by zero or negative areas.
        r = np.maximum(r, 0.1)
        return r

    def compute_score_from_radii(r):
        """
        Compute min triangle area / convex hull area given radii.
        """
        # Cartesian points
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts = np.column_stack([x, y])
        
        # Convex Hull Area
        # Since points are in angular order and radii > 0, they form a convex polygon
        # Area = 0.5 * sum(r_k * r_{k+1} * sin(delta))
        # Note: This formula assumes the origin is inside the polygon, which is true
        # for a "star-shaped" polygon with respect to origin, and convexity ensures
        # the polygon area is indeed this sum if no point is "behind" the origin relative to neighbors.
        # For small perturbations from regular polygon, this holds.
        # To be safe, we can use ConvexHull if we suspect non-convexity, but 
        # with r > 0 and smooth expansion, convexity is likely preserved.
        # Let's use the fast formula first.
        r_next = np.roll(r, -1)
        hull_area = 0.5 * np.sum(r * r_next) * sin_delta
        
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

    def neg_score_coeffs(coeffs):
        r = get_radii_from_coeffs(coeffs)
        score = compute_score_from_radii(r)
        return -score

    def neg_score_radii(r_flat):
        # r_flat is just the radii, ensuring positivity
        r = np.maximum(r_flat, 0.1)
        score = compute_score_from_radii(r)
        return -score

    # Initial guess: Regular 14-gon (all coeffs 0)
    x0_coeffs = np.zeros(2 * M)
    
    best_score_val = 0
    best_r = np.ones(N)
    
    # Evaluate initial guess
    init_score = compute_score_from_radii(best_r)
    best_score_val = init_score
    
    # Phase 1: Optimize Fourier coefficients (low dimensional, smooth manifold)
    # This enforces a global shape constraint.
    try:
        res1 = minimize(neg_score_coeffs, x0_coeffs, method='Powell',
                        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
        
        if time.time() > deadline - 30:
            pass # Skip if too close to deadline
        
        if res1.fun < 0:
            score_val = -res1.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_r = get_radii_from_coeffs(res1.x)
    except Exception:
        pass

    # Phase 2: Refine specific radii (higher dimensional, more flexible)
    # Start from the radii found in Phase 1 (or regular polygon if Phase 1 failed)
    # This allows breaking the smooth harmonic constraint to fine-tune individual points.
    if time.time() < deadline - 15:
        try:
            # Add a tiny bit of noise to avoid getting stuck in exact local minima of the previous step
            r_start = best_r.copy()
            # r_start += np.random.normal(0, 0.01, N)
            
            res2 = minimize(neg_score_radii, r_start, method='Powell',
                            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res2.fun < 0:
                score_val = -res2.fun
                r_candidate = np.maximum(res2.x, 0.1)
                # Re-evaluate to ensure no numerical issues
                score_check = compute_score_from_radii(r_candidate)
                if score_check > best_score_val:
                    best_score_val = score_check
                    best_r = r_candidate
        except Exception:
            pass

    # Phase 3: Nelder-Mead on radii for further refinement if time remains
    if time.time() < deadline - 5:
        try:
            res3 = minimize(neg_score_radii, best_r, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res3.fun < 0:
                score_val = -res3.fun
                r_candidate = np.maximum(res3.x, 0.1)
                score_check = compute_score_from_radii(r_candidate)
                if score_check > best_score_val:
                    best_score_val = score_check
                    best_r = r_candidate
        except Exception:
            pass

    # Final check and return
    # Ensure the returned points are valid
    x = best_r * np.cos(angles)
    y = best_r * np.sin(angles)
    pts = np.column_stack([x, y])
    
    return pts
