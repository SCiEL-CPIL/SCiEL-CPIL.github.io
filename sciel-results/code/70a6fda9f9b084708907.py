# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    
    Strategy:
      - Parameterize vertices by radial distance r(theta) = 1 + sum c_k * cos(k*theta)
        using Fourier modes 1, 2, and 3 (M=3) to maintain smoothness and symmetry.
      - Use a smooth soft-min surrogate for the score to allow gradient-based optimization.
      - Use Multi-Start SLSQP optimization with explicit convexity constraints.
      - Deadline guard: return best valid polygon before 625s.
    """
    start_time = time.time()
    deadline = 625.0

    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Use 3 Fourier modes (1, 2, 3) based on observed improvement in similar tasks
    M = 3
    n_params = M
    
    # Precompute cos(k*theta) for k=1..M
    cos_terms = np.zeros((n, M))
    for m in range(1, M + 1):
        cos_terms[:, m - 1] = np.cos(m * base_angles)

    def params_to_polygon(params):
        """Convert radial coefficients to (20, 2) polygon."""
        r = 1.0 + cos_terms @ params
        x = r * np.cos(base_angles)
        y = r * np.sin(base_angles)
        return np.stack([x, y], axis=1)

    def is_convex(polygon):
        """Check strict convexity: all cross products same sign, nonzero."""
        n_pts = len(polygon)
        cross_vals = np.zeros(n_pts)
        for i in range(n_pts):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n_pts]
            p2 = polygon[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross_vals[i] = v1[0] * v2[1] - v1[1] * v2[0]
        if np.any(np.abs(cross_vals) < 1e-12):
            return False
        if not (np.all(cross_vals > 0) or np.all(cross_vals < 0)):
            return False
        return True

    def compute_scores(polygon):
        """Vectorized per-vertex score computation. Returns array of 20 scores."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_c = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_c)
        if mean_dist <= 0.01:
            return np.zeros(n)
        normalized = centered / mean_dist

        if not is_convex(normalized):
            return np.zeros(n)

        # Check no repeated points
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                    return np.zeros(n)

        scores = np.ones(n)
        for v_idx in range(n):
            v = normalized[v_idx]
            d = np.linalg.norm(normalized - v, axis=1)
            d[v_idx] = np.inf
            d_sorted = np.sort(d)

            if d_sorted[0] < 0.05:
                scores[v_idx] = 0.0
                continue

            gaps = np.diff(d_sorted)
            k = int(np.argmin(gaps))
            D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
            if D < 0.1:
                scores[v_idx] = 0.0
                continue

            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k - 1:k + 3]

            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            scores[v_idx] = min(1.0, 1.0 / raw) if raw > 0 else 0.0

        return scores

    def compute_min_score(polygon):
        """Return the minimum vertex score (the metric)."""
        scores = compute_scores(polygon)
        return float(np.min(scores))

    def soft_min(scores, beta=100.0):
        """Soft minimum via log-sum-exp for smooth surrogate."""
        scores = np.asarray(scores)
        if np.any(scores <= 0):
            return 0.0
        shifted = -beta * scores
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.mean(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def objective(params):
        """Negative soft score for minimization."""
        if time.time() - start_time > deadline:
            return 10.0
        try:
            poly = params_to_polygon(params)
            # Quick convexity check for early exit
            if not is_convex(poly):
                return 5.0
            
            centroid = np.mean(poly, axis=0)
            centered = poly - centroid
            dists_c = np.linalg.norm(centered, axis=1)
            mean_dist = np.mean(dists_c)
            if mean_dist <= 0.01:
                return 10.0
            normalized = centered / mean_dist
            
            # Compute scores for soft min
            scores = np.ones(n)
            for v_idx in range(n):
                v = normalized[v_idx]
                d = np.linalg.norm(normalized - v, axis=1)
                d[v_idx] = np.inf
                d_sorted = np.sort(d)

                if d_sorted[0] < 0.05:
                    scores[v_idx] = 0.0
                    continue

                gaps = np.diff(d_sorted)
                k = int(np.argmin(gaps))
                D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
                if D < 0.1:
                    scores[v_idx] = 0.0
                    continue

                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k - 1:k + 3]

                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                scores[v_idx] = min(1.0, 1.0 / raw) if raw > 0 else 0.0

            sm = soft_min(scores, beta=100.0)
            return -sm if sm > 0 else 5.0
        except Exception:
            return 5.0

    def convexity_constraints(params):
        """
        Constraints for SLSQP: cross products must be positive.
        We use the normalized polygon for consistency, but SLSQP works on params.
        We return the cross products themselves, requiring them to be > 0.
        """
        poly = params_to_polygon(params)
        centroid = np.mean(poly, axis=0)
        normalized = (poly - centroid) # Scaling doesn't change sign of cross product
        cross_vals = np.zeros(n)
        for i in range(n):
            p0 = normalized[i]
            p1 = normalized[(i + 1) % n]
            p2 = normalized[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross_vals[i] = v1[0] * v2[1] - v1[1] * v2[0]
        return cross_vals

    # Initial: regular polygon
    best_params = np.zeros(n_params)
    best_poly = params_to_polygon(best_params)
    best_score = compute_min_score(best_poly)
    
    # Define bounds for SLSQP
    bounds = [(-0.2, 0.2)] * n_params
    
    # Multi-start SLSQP
    np.random.seed(42)
    n_starts = 20
    start_points = [np.zeros(n_params)]
    for _ in range(n_starts - 1):
        # Random perturbations
        p = np.random.uniform(-0.1, 0.1, n_params)
        start_points.append(p)
        
    # Also try specific symmetric perturbations
    start_points.append(np.array([0.05, -0.02, 0.01]))
    start_points.append(np.array([-0.05, 0.02, -0.01]))
    start_points.append(np.array([0.0, 0.05, 0.0]))
    start_points.append(np.array([0.0, -0.05, 0.0]))

    for sp in start_points:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            # SLSQP with convexity constraints
            # We want cross_vals > 1e-6
            cons = [{'type': 'ineq', 'fun': lambda p: convexity_constraints(p) - 1e-6}]
            
            res = minimize(
                objective, sp,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success:
                sc = compute_min_score(params_to_polygon(res.x))
                if sc > best_score:
                    best_score = sc
                    best_params = res.x.copy()
                    best_poly = params_to_polygon(res.x)
        except Exception:
            pass

    # Final polish with Nelder-Mead on the best found
    if time.time() - start_time < deadline - 5:
        def local_obj(params):
            if time.time() - start_time > deadline - 2:
                return 10.0
            try:
                poly = params_to_polygon(params)
                sc = compute_min_score(poly)
                return -sc if sc > 0 else 5.0
            except Exception:
                return 5.0

        try:
            res = minimize(
                local_obj, best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            sc = -res.fun
            if 0 < sc < 5 and sc > best_score:
                best_score = sc
                best_params = res.x.copy()
                best_poly = params_to_polygon(res.x)
        except Exception:
            pass

    # Final validation
    final_poly = params_to_polygon(best_params)
    final_score = compute_min_score(final_poly)
    if final_score > best_score:
        best_poly = final_poly
        best_score = final_score

    if not is_convex(best_poly):
        best_poly = params_to_polygon(np.zeros(n_params))

    return best_poly

# EVOLVE-BLOCK-END
