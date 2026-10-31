import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
# For a rectangle with perimeter 4, w + h = 2.
# We assume a square 1x1 is a good starting point, but we can also optimize aspect ratio.
# Let's fix w=1, h=1 for stability and simplicity, as it's often optimal for small N.
W = 1.0
H = 1.0

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_result = None
    best_score = -np.inf
    
    # Helper to evaluate a configuration
    def evaluate_circles(circles):
        # circles: array of shape (N, 3) -> [x, y, r]
        # Check validity
        if len(circles) != N:
            return -np.inf
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Boundary checks
        if np.any(rs <= 0):
            return -np.inf
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > W + 1e-6):
            return -np.inf
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > H + 1e-6):
            return -np.inf
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                rad_sum_sq = (rs[i] + rs[j])**2
                if dist_sq < rad_sum_sq - 1e-6:
                    return -np.inf
        
        return np.sum(rs)

    # Objective function for optimization (minimize negative sum of radii + penalty)
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        circles = params.reshape(N, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Penalty for boundary violations
        pen = 0.0
        pen += np.sum(np.maximum(0, -xs + rs)**2)
        pen += np.sum(np.maximum(0, xs - rs - (W - 2*rs))**2) # This is getting messy, simpler:
        # Let's use simple clipping or large penalty
        # Boundary: x in [r, W-r], y in [r, H-r]
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, xs - (W - rs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, ys - (H - rs))**2)
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                rad_sum = rs[i] + rs[j]
                overlap = rad_sum**2 - dist_sq
                if overlap > 0:
                    pen += overlap**2
        
        # Maximize sum of radii -> Minimize -sum(rs)
        return -np.sum(rs) + 100.0 * pen

    def gradient(params):
        # Numerical gradient is slow, but L-BFGS-B can handle it or we can supply analytical.
        # For simplicity and robustness against bugs in analytical grad, we rely on L-BFGS-B's numerical diff if not provided, 
        # but providing analytical is better. However, the penalty function is non-smooth at 0.
        # Let's use a smooth penalty: max(0, x)^2 is smooth.
        # We will let L-BFGS-B use numerical gradient if we don't provide one, but it might be slow.
        # Let's try providing a simple numerical gradient or just use SLSQP with constraints.
        # Given the time constraint, L-BFGS-B with penalty is faster.
        pass

    # Initial guesses
    def init_hexagonal():
        # Hexagonal packing in 1x1 square
        # Radius approx 1 / (2 * sqrt(3) * cols) ... 
        # Let's just create a grid and perturb
        cols = 5
        rows = 5
        xs = []
        ys = []
        rs = []
        # Approximate radius for 25 circles in 1x1 square is ~0.1
        r_est = 0.1
        for i in range(cols):
            for j in range(rows):
                x = (i + 0.5) * (W / cols)
                y = (j + 0.5) * (H / rows)
                # Offset every other row for hex-like
                if j % 2 == 1:
                    x += 0.5 * (W / cols)
                if x < W and y < H:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r_est)
        # Take first N
        init = np.array(list(zip(xs[:N], ys[:N], rs[:N])))
        return init.flatten()

    def init_random():
        # Random positions, small radii
        xs = np.random.uniform(0.1, 0.9, N)
        ys = np.random.uniform(0.1, 0.9, N)
        rs = np.full(N, 0.05)
        return np.array(list(zip(xs, ys, rs))).flatten()

    # Try a few different initializations
    inits = []
    inits.append(init_hexagonal())
    # Add a few random inits
    for _ in range(5):
        inits.append(init_random())

    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Optimization
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-4: # Heuristic for success
                final_params = res.x
                final_circles = final_params.reshape(N, 3)
                
                # Post-process: Ensure strict non-overlap and boundaries by shrinking radii slightly if needed
                # Or just evaluate
                score = evaluate_circles(final_circles)
                
                if score > best_score:
                    best_score = score
                    best_result = final_circles.copy()
                    
        except Exception as e:
            # If optimization fails, ignore
            pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Verify fallback
        if evaluate_circles(best_result) < 0:
             # Shrink more
             scale = 0.9
             best_result[:, 0] = 0.5 + (best_result[:, 0] - 0.5) * scale
             best_result[:, 1] = 0.5 + (best_result[:, 1] - 0.5) * scale
             best_result[:, 2] = best_result[:, 2] * scale

    return best_result
