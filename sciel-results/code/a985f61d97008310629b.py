"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START
def _pack_circles(n=21, deadline=DEADLINE):
    """
    Pack n circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    Returns array of shape (n, 3) with [x, y, r] for each circle.
    """
    start_time = time.time()
    
    # We'll use a coordinate system where the rectangle has width W and height H
    # with 2*(W+H) <= 4, i.e., W+H <= 2.
    # We can normalize by fixing W+H = 2 and optimizing.
    # Actually, let's parametrize: let the rectangle be [0, W] x [0, H] with W + H = 2.
    # Then perimeter = 2(W+H) = 4.
    # We want to maximize sum of radii subject to:
    #   x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Strategy: Use gradient-based optimization.
    # Variables: W (width), and for each circle (x_i, y_i, r_i).
    # Constraint: W + H = 2 => H = 2 - W.
    # So variables: W, x_1..x_n, y_1..y_n, r_1..r_n => total 1 + 3n variables.
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    # Constraints:
    #   r_i > 0
    #   x_i - r_i >= 0 => r_i - x_i <= 0
    #   x_i + r_i <= W => x_i + r_i - W <= 0
    #   y_i - r_i >= 0 => r_i - y_i <= 0
    #   y_i + r_i <= H = 2 - W => y_i + r_i + W - 2 <= 0
    #   (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0 for i < j
    
    # This is a non-convex problem. We'll use SLSQP with multiple random restarts.
    
    n = 21
    dim = 1 + 3 * n  # W + (x, y, r) for each circle
    
    def unpack(v):
        W = v[0]
        H = 2.0 - W
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i, 0] = v[1 + 3*i]
            circles[i, 1] = v[2 + 3*i]
            circles[i, 2] = v[3 + 3*i]
        return W, H, circles
    
    def objective(v):
        W, H, circles = unpack(v)
        # Penalize if W or H out of range
        if W <= 0.01 or H <= 0.01:
            return 1e10
        r = circles[:, 2]
        # Return negative sum of radii (we minimize)
        return -np.sum(r)
    
    def constraints(v):
        W, H, circles = unpack(v)
        if W <= 0.01 or H <= 0.01:
            return [(-1e10, 'ineq')]
        cons = []
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints:
        # r_i - x_i >= 0
        for i in range(n):
            cons.append((r[i] - x[i], 'ineq'))
        # x_i + r_i - W <= 0 => W - x_i - r_i >= 0
        for i in range(n):
            cons.append((W - x[i] - r[i], 'ineq'))
        # r_i - y_i >= 0
        for i in range(n):
            cons.append((r[i] - y[i], 'ineq'))
        # y_i + r_i - H <= 0 => H - y_i - r_i >= 0
        for i in range(n):
            cons.append((H - y[i] - r[i], 'ineq'))
        # r_i > 0: r_i >= 0.001
        for i in range(n):
            cons.append((r[i] - 0.001, 'ineq'))
        
        # Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                cons.append((dist_sq - min_dist**2, 'ineq'))
        
        return cons
    
    def _try_initial(W0, H0, n_circles):
        """Try a specific initial configuration."""
        v0 = np.zeros(dim)
        v0[0] = W0
        H0_val = 2.0 - W0
        # Place circles in a grid
        side = int(math.ceil(math.sqrt(n_circles)))
        for i in range(n_circles):
            row = i // side
            col = i % side
            x = (col + 0.5) * W0 / side
            y = (row + 0.5) * H0_val / side
            r = min(W0 / side, H0_val / side) * 0.45
            v0[1 + 3*i] = x
            v0[2 + 3*i] = y
            v0[3 + 3*i] = r
        return v0
    
    best_value = -np.inf
    best_v = None
    
    # Try a few different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for ar in aspect_ratios:
        if time.time() - start_time > deadline - 5:
            break
        # W/H = ar, W + H = 2 => W(1 + 1/ar) = 2 => W = 2*ar/(ar+1)
        W0 = 2.0 * ar / (ar + 1.0)
        H0 = 2.0 - W0
        if W0 <= 0.01 or H0 <= 0.01:
            continue
        
        v0 = _try_initial(W0, H0, n)
        
        try:
            res = minimize(objective, v0, method='SLSQP', constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_value:
                W, H, circles = unpack(res.x)
                # Verify validity
                valid = True
                for i in range(n):
                    x, y, r = circles[i]
                    if r < 0.0001 or x < r - 1e-9 or x + r > W + 1e-9 or y < r - 1e-9 or y + r > H + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            d2 = (circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2
                            if d2 < (circles[i,2]+circles[j,2])**2 - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    val = np.sum(circles[:, 2])
                    if val > best_value:
                        best_value = val
                        best_v = res.x.copy()
        except Exception:
            pass
    
    if best_v is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(n):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = min(W, H) / side * 0.45
            cells.append([x, y, r])
        return np.array(cells[:n])
    
    W, H, circles = unpack(best_v)
    # Clip tiny negative radii
    circles[:, 2] = np.maximum(circles[:, 2], 0.001)
    # Ensure circles stay in bounds
    for i in range(n):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
    
    return circles[:n]

def construct():
    start_time = time.time()
    try:
        result = _pack_circles(n=21, deadline=DEADLINE)
    except Exception:
        # Fallback
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(21):
            row = i // side
            col = i % side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            r = min(W, H) / side * 0.45
            cells.append([x, y, r])
        result = np.array(cells[:21])
    
    # Final validation and adjustment
    W = np.max(result[:, 0] + result[:, 2])
    H = np.max(result[:, 1] + result[:, 2])
    # Ensure perimeter <= 4
    if 2 * (W + H) > 4.0:
        scale = 2.0 / (W + H)
        result[:, 0] *= scale
        result[:, 1] *= scale
        result[:, 2] *= scale
    
    return result
# EVOLVE-BLOCK-END
