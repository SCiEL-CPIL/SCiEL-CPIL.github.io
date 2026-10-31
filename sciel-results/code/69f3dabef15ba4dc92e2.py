import numpy as np
import time
import random

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    
    # Check boundaries
    for i in range(n):
        if cx[i] < r[i] - 1e-7 or cx[i] > W - r[i] + 1e-7:
            return None
        if cy[i] < r[i] - 1e-7 or cy[i] > H - r[i] + 1e-7:
            return None
        if r[i] < 1e-7:
            return None
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            if d < r[i] + r[j] - 1e-7:
                return None
    return float(np.sum(r))

def _penalty_objective_and_grad(x, W, H, lam):
    """
    Compute penalty objective and analytic gradient.
    Objective: -sum(r) + lam * (sum boundary_violations^2 + sum overlap_violations^2)
    """
    n = len(x)
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] -= 1.0  # Gradient of -sum(r) wrt r is -1
    
    # 1. Boundary penalties
    # Violation: max(0, r - min_dist_to_wall)
    # We use a smooth approximation or just hard clip for gradient? 
    # Let's use standard penalty: if r > dist, add (r-dist)^2.
    # Grad wrt r: 2(r-dist). Grad wrt x: -2(r-dist) * sign.
    
    for i in range(n):
        d_left = cx[i]
        d_right = W - cx[i]
        d_bottom = cy[i]
        d_top = H - cy[i]
        
        # Check each wall
        if r[i] > d_left:
            v = r[i] - d_left
            obj += lam * v * v
            grad[i, 2] += 2.0 * lam * v
            grad[i, 0] -= 2.0 * lam * v
        if r[i] > d_right:
            v = r[i] - d_right
            obj += lam * v * v
            grad[i, 2] += 2.0 * lam * v
            grad[i, 0] += 2.0 * lam * v
        if r[i] > d_bottom:
            v = r[i] - d_bottom
            obj += lam * v * v
            grad[i, 2] += 2.0 * lam * v
            grad[i, 1] -= 2.0 * lam * v
        if r[i] > d_top:
            v = r[i] - d_top
            obj += lam * v * v
            grad[i, 2] += 2.0 * lam * v
            grad[i, 1] += 2.0 * lam * v

    # 2. Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            if d < req:
                v = req - d
                obj += lam * v * v
                # Gradient of (req - d)^2
                # d/dx_i (req - d)^2 = 2(req-d) * (-d/dx_i d) = 2(req-d) * (dx/d)
                if d > 1e-10:
                    factor = 2.0 * lam * v
                    grad[i, 0] += factor * (dx / d)
                    grad[i, 1] += factor * (dy / d)
                    grad[j, 0] -= factor * (dx / d)
                    grad[j, 1] -= factor * (dy / d)
                    # Gradient wrt radii
                    # d/dr_i (req - d)^2 = 2(req-d) * 1
                    grad[i, 2] += factor
                    grad[j, 2] += factor
                else:
                    # If d is 0, push apart randomly or along axes
                    grad[i, 0] += factor * 0.707
                    grad[i, 1] += factor * 0.707
                    grad[j, 0] -= factor * 0.707
                    grad[j, 1] -= factor * 0.707
                    grad[i, 2] += factor
                    grad[j, 2] += factor
                    
    return obj, grad

def _repair_and_clip(x, W, H):
    """Clip positions and radii to be feasible."""
    cx = np.clip(x[:, 0], 0.0, W)
    cy = np.clip(x[:, 1], 0.0, H)
    r = np.maximum(x[:, 2], 1e-6)
    
    # Enforce boundary constraints strictly by reducing r
    for i in range(N):
        max_r = min(cx[i], W - cx[i], cy[i], H - cy[i])
        r[i] = min(r[i], max(max_r, 1e-6))
        
    return np.column_stack([cx, cy, r])

def _gradient_descent_optimize(x0, W, H, max_iter=2000, lr_init=0.01, lam_init=10.0, lam_growth=1.05):
    """
    Perform gradient descent on the penalty objective.
    """
    x = x0.copy()
    x = _repair_and_clip(x, W, H)
    best_x = x.copy()
    best_sum = _evaluate(x, W, H) or 0.0
    
    lr = lr_init
    lam = lam_init
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 2:
            break
            
        obj, grad = _penalty_objective_and_grad(x, W, H, lam)
        
        # Take step
        x_new = x - lr * grad
        
        # Repair
        x_new = _repair_and_clip(x_new, W, H)
        
        # Check improvement in raw sum (only if valid)
        s_new = _evaluate(x_new, W, H)
        if s_new is not None:
            if s_new > best_sum:
                best_sum = s_new
                best_x = x_new.copy()
                
        # Update x if objective decreased (penalty included)
        obj_new, _ = _penalty_objective_and_grad(x_new, W, H, lam)
        if obj_new < obj:
            x = x_new
            # Slowly increase lambda to enforce constraints
            lam *= lam_growth
            # Decay learning rate slightly
            lr *= 0.999
        else:
            # If objective increased, shrink step
            lr *= 0.8
            if lr < 1e-6:
                # Stuck, maybe increase lambda more aggressively or break
                lam *= 1.5
                if lam > 1e6:
                    break

    return best_x, best_sum

def _repulsive_init(W, H, n_steps=100, temp=0.05):
    """
    Initialize using repulsive forces to create a diverse, spread-out packing.
    """
    # Random start
    x = np.zeros((N, 3))
    x[:, 0] = np.random.uniform(0.1*W, 0.9*W, N)
    x[:, 1] = np.random.uniform(0.1*H, 0.9*H, N)
    x[:, 2] = np.random.uniform(0.05, 0.1, N)
    
    x = _repair_and_clip(x, W, H)
    
    # Simple Langevin-like dynamics
    # Force = repulsion from others + attraction to center? No, just repulsion + boundary push
    for step in range(n_steps):
        forces = np.zeros_like(x)
        
        for i in range(N):
            # Boundary forces (soft)
            if x[i, 0] < x[i, 2]:
                forces[i, 0] += (x[i, 2] - x[i, 0])
            if x[i, 0] > W - x[i, 2]:
                forces[i, 0] -= (x[i, 0] - (W - x[i, 2]))
            if x[i, 1] < x[i, 2]:
                forces[i, 1] += (x[i, 2] - x[i, 1])
            if x[i, 1] > H - x[i, 2]:
                forces[i, 1] -= (x[i, 1] - (H - x[i, 2]))
                
            for j in range(N):
                if i == j: continue
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                d = np.hypot(dx, dy)
                req = x[i, 2] + x[j, 2]
                if d < req and d > 1e-6:
                    # Repel
                    f = (req - d) / d
                    forces[i, 0] += dx * f
                    forces[i, 1] += dy * f
                    forces[j, 0] -= dx * f
                    forces[j, 1] -= dy * f
                elif d <= 1e-6:
                    # Random kick
                    forces[i, 0] += np.random.randn() * 0.1
                    forces[i, 1] += np.random.randn() * 0.1
                    forces[j, 0] -= np.random.randn() * 0.1
                    forces[j, 1] -= np.random.randn() * 0.1

        # Apply forces with damping and noise
        x[:, :2] += 0.1 * forces[:, :2]
        x[:, :2] += np.random.randn(N, 2) * temp
        
        x = _repair_and_clip(x, W, H)
        
        # Cool down
        temp *= 0.95
        
    return x

def _grid_init(W, H, cols, rows):
    """Generate a grid layout."""
    x = np.zeros((N, 3))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= N:
                break
            cx = (c + 0.5) * W / cols
            cy = (r + 0.5) * H / rows
            x[idx, 0] = cx
            x[idx, 1] = cy
            # Initial radius: half the min distance to neighbor or wall
            dx = W / cols
            dy = H / rows
            r_est = 0.4 * min(dx, dy)
            # Boundary check
            r_est = min(r_est, cx, W-cx, cy, H-cy)
            x[idx, 2] = max(r_est, 1e-4)
            idx += 1
    # Fill remaining if any
    while idx < N:
        x[idx, 0] = np.random.uniform(0.1*W, 0.9*W)
        x[idx, 1] = np.random.uniform(0.1*H, 0.9*H)
        x[idx, 2] = 0.05
        idx += 1
        
    return _repair_and_clip(x, W, H)

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    # Aspect ratios: W + H = 2
    # Square is 1x1. Let's try 0.8-1.2
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    random.seed(42)
    np.random.seed(42)
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H < 0.5: continue
        
        # Try multiple initializations
        inits = []
        
        # 1. Grid init
        # 21 circles. 5x4 grid = 20. 6x4 = 24.
        # Let's try 5x4 and 6x4 and 4x5
        for cols, rows in [(5, 4), (6, 4), (4, 5), (6, 3)]:
            if cols * rows >= N:
                inits.append(_grid_init(W, H, cols, rows))
        
        # 2. Repulsive init (3 variants)
        for i in range(3):
            inits.append(_repulsive_init(W, H, n_steps=50, temp=0.05 + 0.01*i))
            
        # 3. Random init
        x_rand = np.zeros((N, 3))
        x_rand[:, 0] = np.random.uniform(0.1*W, 0.9*W, N)
        x_rand[:, 1] = np.random.uniform(0.1*H, 0.9*H, N)
        x_rand[:, 2] = 0.08
        inits.append(_repair_and_clip(x_rand, W, H))
        
        # Optimize each init
        for init in inits:
            if time.time() > DEADLINE - 10:
                break
                
            # Run gradient descent
            # Use a relatively low lambda start to allow movement, then increase
            x_opt, s_opt = _gradient_descent_optimize(
                init, W, H, 
                max_iter=1000, 
                lr_init=0.02, 
                lam_init=5.0, 
                lam_growth=1.02
            )
            
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt
                best_W = W
                best_H = H
                
    # Final polish: If we have a good result, run one more high-precision GD
    if best_arr is not None and time.time() < DEADLINE - 5:
        x_polished, s_polished = _gradient_descent_optimize(
            best_arr, best_W, best_H,
            max_iter=500,
            lr_init=0.005,
            lam_init=100.0,
            lam_growth=1.01
        )
        if s_polished > best_sum:
            best_sum = s_polished
            best_arr = x_polished

    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        best_arr = _grid_init(W, H, 5, 5)
        best_W, best_H = W, H
        # Ensure it's valid
        s = _evaluate(best_arr, best_W, best_H)
        if s is None:
            # Force valid
            for i in range(N):
                best_arr[i, 2] = min(best_arr[i, 2], 0.1)
            best_arr = _repair_and_clip(best_arr, best_W, best_H)

    return best_arr

