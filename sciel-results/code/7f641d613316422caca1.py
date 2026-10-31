"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 115.0  # seconds, safety margin before 120s kill

    # --- Grid initialization (seed-like) ---
    side = int(np.ceil(np.sqrt(N)))  # 5
    # We'll use a 5x5 grid but only place N=21 circles.
    # Let's pick the first 21 positions in a 5x5 grid.
    # Actually, a better init for optimization is a dense grid or random.
    # Let's use a grid of 5x5, picking 21 cells.
    # Or better: use a 7x3 grid? 21 = 7*3.
    # Or 5x5 with 21 filled.
    
    # Let's try a 5x5 grid, filling 21 cells.
    # To maximize symmetry, we might want to fill the center first or corners.
    # But for optimization, a good initial guess helps.
    # Let's use a 5x5 grid, skipping the last 4 cells.
    
    # Actually, let's use a more compact initial layout if possible.
    # But the grid is a safe start.
    
    # Let's use 5x5 grid, 21 circles.
    # Positions in unit square [0,1]x[0,1]
    # x = (i+0.5)/5, y = (j+0.5)/5
    # r = 0.5/5 = 0.1
    
    # We will optimize positions (x,y) and aspect ratio (w) with h = 2-w.
    # Perimeter 2(w+h) = 4 => w+h = 2.
    # Let w be the width, h = 2 - w.
    # Circles must be inside [0,w] x [0,h].
    # And disjoint: dist(i,j) >= r_i + r_j.
    
    # Objective: Maximize sum(r_i).
    # This is equivalent to minimizing -sum(r_i).
    # Constraints:
    # 1. x_i >= r_i
    # 2. x_i <= w - r_i
    # 3. y_i >= r_i
    # 4. y_i <= h - r_i
    # 5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # Variables: x_i, y_i, r_i for i=0..N-1, and w.
    # Total vars: 3*N + 1 = 64.
    
    # Let's define the function to evaluate validity and objective.
    
    def evaluate(x):
        # x: array of [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, w]
        w = x[-1]
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            return -np.inf, False
        
        coords = x[:3*N].reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        # Check if any radius is negative
        if np.any(rs < 0):
            return -np.inf, False
        
        # Check boundary constraints
        if np.any(xs < rs) or np.any(xs > w - rs) or np.any(ys < rs) or np.any(ys > h - rs):
            return -np.inf, False
        
        # Check overlap constraints
        # For efficiency, we can check all pairs. N=21, so 210 pairs.
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist2 = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist2 < min_dist*min_dist - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return -np.inf, False
        
        return np.sum(rs), True
    
    def objective(x):
        val, valid = evaluate(x)
        if not valid:
            # Penalize invalid solutions
            # We want to push towards validity.
            # A simple penalty: return a large positive number (since we minimize -sum(rs), 
            # but wait, we want to maximize sum(rs). So minimize -sum(rs).
            # If invalid, we should return a large positive number for the minimizer.
            # Or better, use a penalty method.
            
            # Let's compute the violation magnitude.
            w = x[-1]
            h = 2.0 - w
            if w <= 0 or h <= 0:
                return 1e6
            
            coords = x[:3*N].reshape(N, 3)
            xs = coords[:, 0]
            ys = coords[:, 1]
            rs = coords[:, 2]
            
            pen = 0.0
            # Boundary violations
            pen += np.sum(np.maximum(0, rs - xs))
            pen += np.sum(np.maximum(0, xs - (w - rs)))
            pen += np.sum(np.maximum(0, rs - ys))
            pen += np.sum(np.maximum(0, ys - (h - rs)))
            
            # Overlap violations
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        pen += (min_dist - dist)
            
            # Negative radius
            pen += np.sum(np.maximum(0, -rs))
            
            return 100.0 * pen - np.sum(rs)
        
        return -val  # Minimize negative sum of radii

    def objective_grad(x):
        # Numerical gradient is safer and fast enough for N=64 vars.
        # Or we can implement analytical gradient if needed.
        # Let's stick to numerical for now to ensure correctness, 
        # but it might be slow. 64 vars, ~1000 evals per grad? 
        # 150s limit. Let's try L-BFGS-B with numerical grad.
        pass

    # Initial guess
    # 5x5 grid
    init = np.zeros(3*N + 1)
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            init[k] = (i + 0.5) / 5.0   # x
            init[k+1] = (j + 0.5) / 5.0 # y
            init[k+2] = 0.095           # r (slightly less than 0.1 to be inside)
            k += 3
    init[-1] = 1.0  # w = 1.0, h = 1.0

    best_val = 0.0
    best_x = init.copy()
    
    # Try to improve using L-BFGS-B
    # We need to be careful with the time.
    
    # Let's run a few iterations of L-BFGS-B
    # Or use a simpler gradient descent if L-BFGS-B is too slow with numerical grad.
    
    # Let's try L-BFGS-B with numerical gradient.
    # Maxiter: 500?
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective, 
            init, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False}
        )
        
        if res.success or res.fun < 0: # If it found a valid solution (fun = -sum(r) < 0)
            val, valid = evaluate(res.x)
            if valid and val > best_val:
                best_val = val
                best_x = res.x
    except Exception:
        pass
    
    # If time remains, try another start or more iterations
    if time.time() - start_time < 60:
        # Try a different initial aspect ratio
        # e.g., w=1.5, h=0.5
        init2 = init.copy()
        init2[-1] = 1.5
        # Scale coordinates? No, just let optimizer fix it.
        # But initial positions might be outside [0,1.5]x[0,0.5]
        # So we need to adjust initial positions to be valid for the new aspect ratio.
        # For w=1.5, h=0.5.
        # Let's place them in a 7x3 grid.
        init2[:3*N] = 0
        k = 0
        for i in range(7):
            for j in range(3):
                if k >= N:
                    break
                init2[k] = (i + 0.5) / 7.0 * 1.5   # x in [0, 1.5]
                init2[k+1] = (j + 0.5) / 3.0 * 0.5 # y in [0, 0.5]
                init2[k+2] = 0.1                    # r
                k += 3
        init2[-1] = 1.5
        
        try:
            res2 = minimize(
                objective, 
                init2, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False}
            )
            
            if res2.success or res2.fun < 0:
                val, valid = evaluate(res2.x)
                if valid and val > best_val:
                    best_val = val
                    best_x = res2.x
        except Exception:
            pass

    # Final check and return
    # Ensure the best_x is valid. If not, fall back to the seed grid.
    val, valid = evaluate(best_x)
    if not valid or val <= 0:
        # Fall back to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output: array of shape (n, 3)
    # The task says "bounding rectangle has perimeter at most 4".
    # Our optimization ensures circles are inside [0,w]x[0,h] with w+h=2.
    # So the bounding box of the *rectangles* is w x h.
    # But the *circles* might not touch the boundaries.
    # The task says "bounding rectangle has perimeter at most 4".
    # Usually this means the rectangle containing the circles.
    # If the circles are strictly inside [0,w]x[0,h], the bounding box of the circles
    # is smaller than w x h, so perimeter is smaller than 4.
    # So we are safe.
    
    # However, the evaluator might check if the circles fit in a rectangle of perimeter 4.
    # Our construction guarantees they fit in a rectangle of perimeter 4 (w+h=2).
    # So we just return the circles.
    
    return best_x[:3*N].reshape(N, 3)
# EVOLVE-BLOCK-END
