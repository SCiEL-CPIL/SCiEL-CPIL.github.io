import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n = 8
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(pairs)

    def compute_diameter(points):
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_d:
                    max_d = d
        return max_d

    def score_points(points):
        d_max = compute_diameter(points)
        if d_max < 1e-12:
            return 0.0
        scale = 2.0 / d_max
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j]) * scale
                if d < 1e-12:
                    return 0.0
                log_prod += math.log(d)
        log_score = log_prod - n * math.log(n)
        if log_score > 700:
            return float('inf')
        return math.exp(log_score)

    # Objective: maximize sum of log(distances) with t as smooth diameter bound
    # We maximize sum(log(d_ij)) - lambda * max(0, d_ij - t)^2 for all pairs
    # Actually, use epigraph: maximize sum(log(d_ij)) subject to d_ij <= t, then scale.
    # Better: parameterize with t and use smooth penalty.

    def neg_log_score_with_t(x):
        """x = [x1,y1,...,x8,y8, t] where t is the diameter variable.
        We maximize sum(log(d_ij)) with constraint d_ij <= t.
        Use smooth penalty for constraint violation."""
        P = x[:2*n].reshape(n, 2)
        t = x[2*n]
        if t < 1e-10:
            return 1e10
        log_sum = 0.0
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(P[i] - P[j])
                if d < 1e-12:
                    return 1e10
                log_sum += math.log(d)
                # Smooth penalty for d > t
                if d > t:
                    penalty += (d - t)**2
        # We want to maximize log_sum, so minimize -log_sum + penalty
        return -log_sum + 100.0 * penalty

    # Alternative: direct approach with diameter constraint via smooth max
    def smooth_max_penalty(x, alpha=100.0):
        """Smooth approximation of max(0, d_ij - 2) using softplus-like."""
        P = x.reshape(n, 2)
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(P[i] - P[j])
                excess = d - 2.0
                if excess > 0:
                    penalty += excess * excess
        return penalty

    def neg_log_score_direct(x):
        P = x.reshape(n, 2)
        log_sum = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(P[i] - P[j])
                if d < 1e-12:
                    return 1e10
                log_sum += math.log(d)
        return -log_sum + 50.0 * smooth_max_penalty(x)

    # Generate diverse initial guesses
    guesses = []

    # 1. Regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(p_oct.flatten())

    # 2. Two squares (one rotated 45 degrees)
    angles_sq1 = 2 * np.pi * np.arange(4) / 4
    angles_sq2 = angles_sq1 + np.pi/4
    p_sq1 = np.column_stack([np.cos(angles_sq1), np.sin(angles_sq1)])
    p_sq2 = np.column_stack([np.cos(angles_sq2), np.sin(angles_sq2)])
    guesses.append(np.vstack([p_sq1, p_sq2]).flatten())

    # 3. Random points
    for seed_val in [42, 7, 99, 2024, 555, 888]:
        np.random.seed(seed_val)
        r = np.random.uniform(0.3, 1.0, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        P = np.column_stack([x, y])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())

    # 4. Points with x-axis symmetry
    for seed_val in [5, 11, 23, 37]:
        np.random.seed(seed_val)
        r = np.random.uniform(0.3, 1.0, 4)
        theta = np.random.uniform(0, np.pi, 4)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        P_upper = np.column_stack([x, y])
        P_lower = np.column_stack([x, -y])
        P = np.vstack([P_upper, P_lower])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())

    # 5. Concentric circles
    for r_inner in [0.2, 0.4, 0.6, 0.8]:
        r_outer = 1.0
        angles_inner = 2 * np.pi * np.arange(4) / 4
        angles_outer = 2 * np.pi * np.arange(4) / 4 + np.pi/4
        p_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        p_outer = r_outer * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        P = np.vstack([p_inner, p_outer])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())

    # 6. Rectangle-based: 4 corners of rectangle + 4 midpoints of edges
    for aspect in [0.5, 0.7, 1.0, 1.4]:
        w = aspect
        h = 1.0
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        mids = np.array([[0, -h/2], [w/2, 0], [0, h/2], [-w/2, 0]])
        P = np.vstack([corners, mids])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())

    # 7. Hexagon + 2 center points
    for center_dist in [0.1, 0.3, 0.5]:
        angles_hex = 2 * np.pi * np.arange(6) / 6
        p_hex = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])
        p_c1 = np.array([center_dist, 0])
        p_c2 = np.array([-center_dist, 0])
        P = np.vstack([p_hex, [p_c1, p_c2]])
        d_max = compute_diameter(P)
        if d_max > 1e-6:
            P = P * (2.0 / d_max)
        guesses.append(P.flatten())

    bounds = [(-2.0, 2.0)] * (2 * n)

    best_points = None
    best_score = -np.inf

    # Optimize each guess
    for idx, p0 in enumerate(guesses):
        if time.time() > deadline - 10.0:
            break

        try:
            # First pass: SLSQP with smooth penalty
            res = minimize(
                neg_log_score_direct,
                p0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )

            if res.x is not None:
                P_opt = res.x.reshape(n, 2)
                # Check distinctness
                min_dist = min(
                    np.linalg.norm(P_opt[i] - P_opt[j])
                    for i in range(n) for j in range(i+1, n)
                )
                if min_dist > 1e-8:
                    sc = score_points(P_opt)
                    if sc > best_score:
                        best_score = sc
                        best_points = P_opt.copy()

                        # Second pass: polish with tighter tolerance
                        if time.time() < deadline - 5.0:
                            res2 = minimize(
                                neg_log_score_direct,
                                P_opt.flatten(),
                                method='SLSQP',
                                bounds=bounds,
                                options={'maxiter': 5000, 'ftol': 1e-15}
                            )
                            if res2.x is not None:
                                P_polish = res2.x.reshape(n, 2)
                                min_dist2 = min(
                                    np.linalg.norm(P_polish[i] - P_polish[j])
                                    for i in range(n) for j in range(i+1, n)
                                )
                                if min_dist2 > 1e-8:
                                    sc2 = score_points(P_polish)
                                    if sc2 > best_score:
                                        best_score = sc2
                                        best_points = P_polish.copy()
        except Exception:
            continue

    # Final polish: use the best points and do a very tight optimization
    if best_points is not None and time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                neg_log_score_direct,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-16}
            )
            if res_final.x is not None:
                P_final = res_final.x.reshape(n, 2)
                min_dist_f = min(
                    np.linalg.norm(P_final[i] - P_final[j])
                    for i in range(n) for j in range(i+1, n)
                )
                if min_dist_f > 1e-8:
                    sc_f = score_points(P_final)
                    if sc_f > best_score:
                        best_score = sc_f
                        best_points = P_final.copy()
        except Exception:
            pass

    # Fallback to regular octagon
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return [[float(p[0]), float(p[1])] for p in best_points]
