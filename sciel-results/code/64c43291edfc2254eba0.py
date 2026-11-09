# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in a unit square.
    Uses local optimization (L-BFGS-B) with a penalty method for constraints.
    """
    start_time = time.time()
    timeout = 600
    n = 10
    
    def objective(vars):
        x = vars[0::3]
        y = vars[1::3]
        s = vars[2::3]
        
        # Penalty for boundary violations
        # Square i is [x_i - s_i/2, x_i + s_i/2]
        # Must be in [0, 1]
        # Violation: max(0, s_i/2 - x_i) + max(0, x_i + s_i/2 - 1)
        pen_x = np.sum(np.maximum(0, s/2 - x) + np.maximum(0, x + s/2 - 1))
        pen_y = np.sum(np.maximum(0, s/2 - y) + np.maximum(0, y + s/2 - 1))
        
        # Penalty for overlap
        # Overlap in x: max(0, (s_i + s_j)/2 - |x_i - x_j|)
        # Overlap in y: max(0, (s_i + s_j)/2 - |y_i - y_j|)
        # If both > 0, they overlap.
        # We penalize the sum of squares of these violations.
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                pen_overlap += ox * ox + oy * oy
                
        W = 10000.0
        return -np.sum(s) + W * (pen_x + pen_y + pen_overlap)

    def is_valid(centers, sides):
        tol = 1e-7
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
                return False
            if s <= 0:
                return False
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                t = (s_i + s_j) / 2 - tol
                if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                    return False
        return True

    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Generate initial guesses
    guesses = []
    
    # Guess 1: 3x3 grid + 1 small square (shrunken to fit)
    # 3x3 of 0.3 squares, placed at [0.1, 1.0] x [0.1, 1.0]
    # 1 square of 0.1 at [0, 0.1] x [0, 0.1]
    s_large = 0.3
    centers_g1 = np.zeros((n, 2))
    sides_g1 = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        centers_g1[i, 0] = 0.1 + (c + 0.5) * s_large
        centers_g1[i, 1] = 0.1 + (r + 0.5) * s_large
        sides_g1[i] = s_large
    s_small = 0.1
    centers_g1[9, 0] = s_small / 2
    centers_g1[9, 1] = s_small / 2
    sides_g1[9] = s_small
    guesses.append(np.concatenate([centers_g1[:, 0], centers_g1[:, 1], sides_g1]))
    
    # Guess 2: Uniform size 0.31, random positions
    for _ in range(50):
        s_rand = np.full(n, 0.31)
        x_rand = np.random.uniform(0.2, 0.8, n)
        y_rand = np.random.uniform(0.2, 0.8, n)
        guesses.append(np.concatenate([x_rand, y_rand, s_rand]))
        
    # Guess 3: Uniform size 0.3, random positions
    for _ in range(50):
        s_rand = np.full(n, 0.3)
        x_rand = np.random.uniform(0.2, 0.8, n)
        y_rand = np.random.uniform(0.2, 0.8, n)
        guesses.append(np.concatenate([x_rand, y_rand, s_rand]))

    for guess in guesses:
        if time.time() - start_time > timeout - 10:
            break
        try:
            res = minimize(objective, guess, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                x = res.x[0::3]
                y = res.x[1::3]
                s = res.x[2::3]
                centers = np.stack([x, y], axis=1)
                sides = s
                if is_valid(centers, sides):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
                        best_sides = sides
        except Exception:
            pass

    if best_centers is None:
        # Fallback to 3x3+1
        return centers_g1, sides_g1

    return best_centers, best_sides
# EVOLVE-BLOCK-END
