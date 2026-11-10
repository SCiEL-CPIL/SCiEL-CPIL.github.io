import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = 600.0  # safety margin from 630s limit

    # Precompute pair indices for pdist
    pair_i = np.array([i for i in range(n) for j in range(i + 1, n)])
    pair_j = np.array([j for i in range(n) for j in range(i + 1, n)])
    npairs = len(pair_i)

    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        pts = pts - np.mean(pts, axis=0)
        dm = np.max(pdist(pts))
        if dm > 1e-15:
            pts = pts / dm
        return pts

    def distances_from_x(x):
        pts = x.reshape(-1, 2)
        di = pts[pair_i]
        dj = pts[pair_j]
        return np.linalg.norm(di - dj, axis=1)

    def smooth_max_log(dists, scale):
        """Smooth approximation of log(max(dists)) using log-sum-exp."""
        # log(max) ≈ (1/scale) * log(sum(exp(scale*d)))
        scaled = scale * dists
        max_s = np.max(scaled)
        return max_s / scale + np.log(np.sum(np.exp(scaled - max_s))) / scale

    def smooth_min_log(dists, scale):
        """Smooth approximation of log(min(dists)) using log-sum-exp of negatives."""
        # log(min) = -log(max(-d)) ≈ -(1/scale)*log(sum(exp(-scale*d)))
        scaled = -scale * dists
        max_s = np.max(scaled)
        return -(max_s / scale + np.log(np.sum(np.exp(scaled - max_s))) / scale)

    def objective(x, scale=20.0):
        """Minimize -log(d_min/d_max) = -log(d_min) + log(d_max) using smooth surrogates."""
        dists = distances_from_x(x)
        if np.min(dists) < 1e-12:
            return 1e6
        obj = -smooth_min_log(dists, scale) + smooth_max_log(dists, scale)
        return obj

    def gradient(x, scale=20.0):
        """Gradient of objective w.r.t. x."""
        pts = x.reshape(-1, 2)
        di = pts[pair_i]
        dj = pts[pair_j]
        vecs = di - dj
        dists = np.linalg.norm(vecs, axis=1)
        safe_dists = np.maximum(dists, 1e-12)

        # Gradient of smooth_max_log:
        # d/dx log-sum-exp(scale*d)/scale
        # = (1/scale) * sum_k (scale * exp(scale*d_k)/sum(exp(scale*d_k))) * d d_k/dx
        # = sum_k w_k * d d_k/dx where w_k = exp(scale*d_k)/sum(exp(scale*d_k))
        scaled_max = scale * dists
        max_sm = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_sm)
        w_max = exp_max / np.sum(exp_max)  # softmax weights

        # Gradient of -smooth_min_log:
        # = d/dx log-sum-exp(-scale*d)/scale
        # = sum_k v_k * (-d d_k/dx) where v_k = exp(-scale*d_k)/sum(exp(-scale*d_k))
        scaled_min = -scale * dists
        max_sm2 = np.max(scaled_min)
        exp_min = np.exp(scaled_min - max_sm2)
        v_min = exp_min / np.sum(exp_min)  # softmax weights for min

        # d d_k / d x_i = (x_i - x_j) / d_k
        # d d_k / d x_j = (x_j - x_i) / d_k

        grad = np.zeros_like(x)
        # For each pair k, contribution to gradient:
        # from max term: w_k * (vecs_k / d_k) to x_i, -w_k * (vecs_k / d_k) to x_j
        # from min term: -v_k * (vecs_k / d_k) to x_i, +v_k * (vecs_k / d_k) to x_j
        # Total coefficient for x_i: (w_k - v_k) / d_k
        # Total coefficient for x_j: -(w_k - v_k) / d_k

        coeffs = (w_max - v_min) / safe_dists
        for k in range(npairs):
            c = coeffs[k]
            if abs(c) > 1e-15:
                grad[2 * pair_i[k]:2 * pair_i[k] + 2] += c * vecs[k]
                grad[2 * pair_j[k]:2 * pair_j[k] + 2] -= c * vecs[k]

        return grad

    def local_optimize(x0, scale=20.0, max_iter=300):
        """Run L-BFGS-B from x0 with smooth objective."""
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=lambda x: gradient(x, scale),
                           args=(scale,), options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                                                   'maxls': 50})
            return res.x
        except Exception:
            return x0

    def refine_with_bisection(pts, max_iter=50, tol=1e-8):
        """
        Bisection on minimum distance constraint:
        Find max d_min such that there exists a configuration with all distances >= d_min
        and d_max = 1 (after normalization).
        """
        # We work in normalized space where d_max = 1
        # Binary search on d_min
        lo = 0.0
        hi = 1.0  # d_min <= d_max = 1 always

        # First, find a feasible d_min by checking current
        cur_dists = pdist(pts)
        cur_dmin = np.min(cur_dists)
        cur_dmax = np.max(cur_dists)
        if cur_dmax > 0:
            lo = cur_dmin / cur_dmax  # current ratio
            hi = min(1.0, lo + 0.1)  # upper bound

        best_pts = pts.copy()
        best_ratio = lo

        for _ in range(max_iter):
            if time.time() - start_time > deadline - 30:
                break
            mid = (lo + hi) / 2
            # Try to find configuration with d_min >= mid * d_max and d_max = 1
            # i.e., all distances in [mid, 1]
            # Minimize sum of violations: sum(max(0, mid - d_k)^2) + sum(max(0, d_k - 1)^2)
            # But this is non-smooth. Instead, use the smooth approach:
            # Maximize d_min subject to d_max <= 1
            # Equivalent to: minimize -log(d_min) subject to all d_k <= 1
            # Use penalty method

            def penalty_obj(x):
                d = distances_from_x(x)
                # Penalty for distances below mid
                below = np.maximum(0, mid - d)
                # Penalty for distances above 1
                above = np.maximum(0, d - 1.0)
                # Smooth max of d
                sm = 10.0
                scaled = sm * d
                mx = np.max(scaled)
                log_max = mx / sm + np.log(np.sum(np.exp(scaled - mx))) / sm
                # We want to maximize d_min, so minimize -log(d_min) + penalty
                # Use smooth min
                scaled_neg = -sm * d
                mx2 = np.max(scaled_neg)
                log_min = -(mx2 / sm + np.log(np.sum(np.exp(scaled_neg - mx2))) / sm)
                pen = 100.0 * (np.sum(below**2) + np.sum(above**2))
                return -log_min + pen

            def penalty_grad(x):
                pts_x = x.reshape(-1, 2)
                di = pts_x[pair_i]
                dj = pts_x[pair_j]
                vecs = di - dj
                d = np.linalg.norm(vecs, axis=1)
                safe_d = np.maximum(d, 1e-12)

                sm = 10.0
                scaled_neg = -sm * d
                mx2 = np.max(scaled_neg)
                exp_neg = np.exp(scaled_neg - mx2)
                v = exp_neg / np.sum(exp_neg)

                below = np.maximum(0, mid - d)
                above = np.maximum(0, d - 1.0)

                # Gradient of -log_min: same as before but with v weights
                # d(-log_min)/dx_i = -sum_k v_k * d(d_k)/dx_i = -sum_k v_k * (x_i - x_j)/d_k
                # Gradient of penalty:
                # d(sum below^2)/dx_i = 2 * sum_k below_k * (-1) * (x_i - x_j)/d_k for below_k > 0
                # d(sum above^2)/dx_i = 2 * sum_k above_k * (x_i - x_j)/d_k for above_k > 0

                grad = np.zeros_like(x)
                # For -log_min part:
                coeff_min = -v / safe_d
                for k in range(npairs):
                    c = coeff_min[k]
                    if abs(c) > 1e-15:
                        grad[2 * pair_i[k]:2 * pair_i[k] + 2] += c * vecs[k]
                        grad[2 * pair_j[k]:2 * pair_j[k] + 2] -= c * vecs[k]

                # For penalty part:
                coeff_pen = 200.0 * (-below + above) / safe_d
                for k in range(npairs):
                    c = coeff_pen[k]
                    if abs(c) > 1e-15:
                        grad[2 * pair_i[k]:2 * pair_i[k] + 2] += c * vecs[k]
                        grad[2 * pair_j[k]:2 * pair_j[k] + 2] -= c * vecs[k]

                return grad

            x0 = pts.flatten()
            try:
                res = minimize(penalty_obj, x0, method='L-BFGS-B', jac=penalty_grad,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                new_pts = res.x.reshape(-1, 2)
                new_dists = pdist(new_pts)
                new_dmin = np.min(new_dists)
                new_dmax = np.max(new_dists)
                if new_dmax > 0:
                    new_ratio = new_dmin / new_dmax
                    if new_ratio >= mid - tol:
                        lo = mid
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_pts = new_pts.copy()
                    else:
                        hi = mid
            except Exception:
                hi = mid

        return best_pts, best_ratio

    # Generate diverse initializations
    np.random.seed(42)
    initializations = []

    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Two-ring (inner 6, outer 10)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.45 * np.cos(inner_angles), 0.45 * np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initializations.append(np.vstack([inner_pts, outer_pts]))

    # 3. Three-ring
    r1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    r2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    r3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    initializations.append(np.vstack([
        np.column_stack([0.3 * np.cos(r1), 0.3 * np.sin(r1)]),
        np.column_stack([0.65 * np.cos(r2), 0.65 * np.sin(r2)]),
        np.column_stack([np.cos(r3), np.sin(r3)])
    ]))

    # 4. Hexagonal lattice subset
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            hex_pts.append([i + 0.5 * j, j * np.sqrt(3) / 2])
    hex_pts = np.array(hex_pts)
    dc = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dc)[:n]
    initializations.append(hex_pts[idx])

    # 5. Grid 4x4
    gx, gy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    initializations.append(np.column_stack([gx.ravel(), gy.ravel()]))

    # 6. Perturbed circles
    for _ in range(6):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.randn(n) * 0.2
        radii = 1 + 0.2 * np.random.randn(n)
        initializations.append(np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii]))

    # 7. Random
    for _ in range(8):
        initializations.append(np.random.randn(n, 2))

    # 8. Two-ring with different ratios
    for r_inner in [0.3, 0.5, 0.6]:
        ia = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        oa = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
        initializations.append(np.vstack([
            np.column_stack([r_inner * np.cos(ia), r_inner * np.sin(ia)]),
            np.column_stack([np.cos(oa), np.sin(oa)])
        ]))

    # Process initializations
    best_score = 0.0
    best_points = None

    for init in initializations:
        if time.time() - start_time > deadline - 40:
            break

        pts = normalize_pts(init)
        score = compute_score(pts)

        if score > best_score:
            best_score = score
            best_points = pts.copy()

        # Phase 1: Smooth gradient optimization with multiple scales
        x = pts.flatten()
        for scale in [10.0, 20.0, 50.0, 100.0]:
            if time.time() - start_time > deadline - 30:
                break
            x = local_optimize(x, scale=scale, max_iter=200)
            pts = normalize_pts(x.reshape(-1, 2))
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()

        # Phase 2: Bisection refinement
        if time.time() - start_time > deadline - 20:
            continue
        pts_b, ratio_b = refine_with_bisection(pts, max_iter=30)
        score_b = compute_score(pts_b)
        if score_b > best_score:
            best_score = score_b
            best_points = pts_b.copy()

    # Final polish from best point
    if best_points is not None and time.time() - start_time < deadline - 15:
        x = best_points.flatten()
        for scale in [50.0, 100.0, 200.0]:
            if time.time() - start_time > deadline - 10:
                break
            x = local_optimize(x, scale=scale, max_iter=100)
            pts = normalize_pts(x.reshape(-1, 2))
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()

    if best_points is None:
        best_points = normalize_pts(np.random.randn(n, 2))

    return best_points
