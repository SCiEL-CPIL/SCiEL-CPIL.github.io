import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10():
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin for 600s limit

    tol = 1e-7

    def is_valid(cx, cy, s):
        """Check if the configuration is valid."""
        if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy)) and np.all(np.isfinite(s))):
            return False
        if np.any(s <= 0):
            return False
        # Bounds check
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        # Overlap check
        cx_col = cx[:, np.newaxis]
        cy_col = cy[:, np.newaxis]
        s_col = s[:, np.newaxis]
        dx = np.abs(cx_col - cx_col.T)
        dy = np.abs(cy_col - cy_col.T)
        t = (s_col + s_col.T) / 2.0 - tol
        mask = ~np.eye(n, dtype=bool)
        overlaps = (dx < t) & (dy < t) & mask
        if np.any(overlaps):
            return False
        return True

    def objective(x):
        """
        x layout: [s_0..s_9, cx_0..cx_9, cy_0..cy_9]  (30 vars)
        We want to maximize sum(s). So minimize -sum(s) + penalty.
        """
        s = x[:n]
        cx = x[n:2*n]
        cy = x[2*n:3*n]

        # Penalty for going out of bounds
        # We use a soft penalty: if edge < 0 or > 1, add large penalty
        lower_violation = np.maximum(0, -(cx - s/2)) + np.maximum(0, -(cy - s/2))
        upper_violation = np.maximum(0, (cx + s/2) - 1) + np.maximum(0, (cy + s/2) - 1)
        bound_penalty = 1e4 * (np.sum(lower_violation) + np.sum(upper_violation))

        # Penalty for overlap
        # For each pair (i,j), i<j:
        # overlap_x = max(0, (s_i+s_j)/2 - |cx_i - cx_j|)
        # overlap_y = max(0, (s_i+s_j)/2 - |cy_i - cy_j|)
        # overlap_volume = overlap_x * overlap_y
        # We sum these and apply a penalty.
        # To make it differentiable, we use a smooth max: softplus(beta * z) / beta
        # But simple max(0, z) is non-smooth. L-BFGS-B can handle non-smoothness
        # via finite differences, but it's better to smooth it.
        # Let's use: smooth_max(z) = log(1 + exp(beta*z))/beta for z>0, else 0?
        # Actually, the standard smooth max is: (z + sqrt(z^2 + eps))/2 for z, but that's for max(0,z) approximation?
        # No, max(0, z) can be approximated by (z + sqrt(z^2 + eps))/2.
        # Let's use eps = 1e-6 for smoothness.
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # overlap if dx < t and dy < t
                # amount of overlap in x: t - dx (if positive)
                # amount of overlap in y: t - dy (if positive)
                # We want to penalize the area of overlap.
                # ox = max(0, t - dx)
                # oy = max(0, t - dy)
                # penalty = ox * oy
                # Smooth max(0, z) = (z + sqrt(z^2 + eps))/2
                eps = 1e-6
                ox = (t - dx + np.sqrt((t - dx)**2 + eps)) / 2.0
                oy = (t - dy + np.sqrt((t - dy)**2 + eps)) / 2.0
                overlap_penalty += ox * oy

        overlap_penalty *= 1e4  # High penalty to enforce separation

        # Objective: minimize -sum(s) + bound_penalty + overlap_penalty
        return -np.sum(s) + bound_penalty + overlap_penalty

    def constraint_bounds(x):
        """
        Define bounds for L-BFGS-B.
        s_i in [0.01, 1.0]
        cx_i in [s_i/2, 1 - s_i/2] -> but s_i is a variable, so this is coupled.
        L-BFGS-B requires fixed bounds.
        We can set loose bounds:
        s in [0.01, 1.0]
        cx in [0, 1]
        cy in [0, 1]
        And rely on the penalty to keep them within [s/2, 1-s/2].
        """
        pass

    bounds = [(0.01, 1.0)] * n + [(0.0, 1.0)] * (2 * n)

    best_score = -np.inf
    best_x = None

    # Generate multiple initializations
    inits = []

    # 1. Grid initialization (2x5)
    s0 = np.full(10, 0.2)
    cx0 = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    cy0 = np.array([0.1]*5 + [0.3]*5)
    inits.append(np.concatenate([s0, cx0, cy0]))

    # 2. "2 large + 8 small" initialization
    # 2 squares of 0.5, 8 squares of 0.25
    s1 = np.array([0.5, 0.5] + [0.25]*8)
    cx1 = np.array([0.25, 0.25, 
                    0.625, 0.875, 0.625, 0.875, 
                    0.625, 0.875, 0.625, 0.875])
    cy1 = np.array([0.25, 0.75, 
                    0.125, 0.125, 0.375, 0.375, 
                    0.625, 0.625, 0.875, 0.875])
    inits.append(np.concatenate([s1, cx1, cy1]))

    # 3. Random initializations
    np.random.seed(42)
    for k in range(5):
        s_r = np.random.uniform(0.1, 0.4, 10)
        cx_r = np.random.uniform(0.05, 0.95, 10)
        cy_r = np.random.uniform(0.05, 0.95, 10)
        inits.append(np.concatenate([s_r, cx_r, cy_r]))

    for x0 in inits:
        if time.time() > deadline - 30:
            break

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-10, 'maxls': 50})
            
            if res.success or res.status == 1:  # status 1 means max iterations
                x_opt = res.x
                s_opt = x_opt[:n]
                cx_opt = x_opt[n:2*n]
                cy_opt = x_opt[2*n:3*n]

                # Check validity
                if is_valid(cx_opt, cy_opt, s_opt):
                    score = np.sum(s_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
        except Exception:
            continue

    # If no valid result from optimization, use fallback
    if best_x is None:
        # Fallback: simple grid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        centers = np.stack([cx, cy], axis=1)
        return centers, s

    s_final = best_x[:n]
    cx_final = best_x[n:2*n]
    cy_final = best_x[2*n:3*n]

    # Final validity check
    if not is_valid(cx_final, cy_final, s_final):
        # Fallback
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx_final = (cols + 0.5) * side
        cy_final = (rows + 0.5) * side
        s_final = np.full(n, side)

    centers = np.stack([cx_final, cy_final], axis=1)
    return centers, s_final
