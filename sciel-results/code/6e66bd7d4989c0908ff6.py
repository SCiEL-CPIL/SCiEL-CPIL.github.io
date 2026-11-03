import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def generate_grid_init(w_init, strategy=0):
        h_init = 2.0 - w_init
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        
        if strategy == 0:
            # Standard grid
            for i in range(side):
                for j in range(side):
                    if len(xs) >= N:
                        break
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.9
                    xs.append(cx)
                    ys.append(cy)
                    rs.append(r)
        elif strategy == 1:
            # Hexagonal-ish packing approximation
            # This is a heuristic, not a true hex pack, but tries to stagger rows
            rows = int(np.ceil(N / 5))
            cols = 5
            for i in range(rows):
                for j in range(cols):
                    if len(xs) >= N:
                        break
                    # Stagger odd rows
                    offset = 0.0 if i % 2 == 0 else 0.5
                    cx = (j + offset + 0.5) / cols * w_init
                    cy = (i + 0.5) / rows * h_init
                    # Keep within bounds
                    cx = np.clip(cx, 0.1, w_init - 0.1)
                    cy = np.clip(cy, 0.1, h_init - 0.1)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.8
                    xs.append(cx)
                    ys.append(cy)
                    rs.append(r)
        elif strategy == 2:
            # Randomized jitter on grid
            for i in range(side):
                for j in range(side):
                    if len(xs) >= N:
                        break
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    # Add small random jitter
                    cx += (np.random.rand() - 0.5) * 0.1 * w_init / side
                    cy += (np.random.rand() - 0.5) * 0.1 * h_init / side
                    cx = np.clip(cx, 0.05, w_init - 0.05)
                    cy = np.clip(cy, 0.05, h_init - 0.05)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.9
                    xs.append(cx)
                    ys.append(cy)
                    rs.append(r)

        while len(xs) < N:
            xs.append(w_init/2)
            ys.append(h_init/2)
            rs.append(0.1)
            
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        rs = np.array(rs[:N])
        return xs, ys, rs

    def objective(v):
        w = v[0]
        if w <= 0.01 or w >= 1.99:
            return 1e10
        h = 2.0 - w
        
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        if np.any(rs < 0):
            return 1e10
            
        # Boundary penalties
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        return -np.sum(rs) + 100.0 * pen

    def polish_solution(x0, w_init, max_iters=1000):
        """
        Local polish using L-BFGS-B with tighter tolerances.
        """
        h_init = 2.0 - w_init
        xs = x0[1:N+1]
        ys = x0[N+1:2*N+1]
        rs = x0[2*N+1:3*N+1]
        
        # Initial cleanup: ensure bounds are respected to avoid large initial penalties
        for i in range(N):
            rs[i] = min(rs[i], xs[i], w_init - xs[i], ys[i], h_init - ys[i])
            if rs[i] < 0:
                rs[i] = 0
            
        x_polish = np.concatenate(([w_init], xs, ys, rs))
        
        # Use L-BFGS-B for polish
        # We fix w for a moment to stabilize, or allow it to move slightly?
        # Let's allow w to move but keep it close to w_init.
        
        res = minimize(objective, x_polish, method='L-BFGS-B', 
                       options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-10})
        
        return res.x, res.fun

    def resolve_collisions(circles, w, h, iters=200):
        """
        Physics-based collision resolution to fix any remaining violations
        and potentially improve packing by allowing slight radius adjustments?
        No, let's just fix positions to ensure validity, then shrink radii if needed.
        """
        circles = circles.copy()
        for _ in range(iters):
            if time.time() > deadline:
                break
            moved = False
            # Resolve boundary violations
            for i in range(N):
                x, y, r = circles[i]
                if x - r < 0:
                    circles[i, 0] = r
                    moved = True
                if x + r > w:
                    circles[i, 0] = w - r
                    moved = True
                if y - r < 0:
                    circles[i, 1] = r
                    moved = True
                if y + r > h:
                    circles[i, 1] = h - r
                    moved = True
            
            # Resolve pairwise collisions
            for i in range(N):
                for j in range(i+1, N):
                    dx = circles[j,0] - circles[i,0]
                    dy = circles[j,1] - circles[i,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i,2] + circles[j,2]
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart equally
                        circles[i,0] -= nx * overlap * 0.5
                        circles[i,1] -= ny * overlap * 0.5
                        circles[j,0] += nx * overlap * 0.5
                        circles[j,1] += ny * overlap * 0.5
                        moved = True
                    elif dist <= 1e-10:
                        # Random small push
                        circles[i,0] -= 0.001
                        circles[j,0] += 0.001
                        moved = True
            
            # Re-check bounds after push
            for i in range(N):
                x, y, r = circles[i]
                if x - r < 0:
                    circles[i, 0] = r
                if x + r > w:
                    circles[i, 0] = w - r
                if y - r < 0:
                    circles[i, 1] = r
                if y + r > h:
                    circles[i, 1] = h - r

            if not moved:
                break
        return circles

    def validate_solution(circles, w, h, tol=1e-6):
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        if np.any(rs < -tol):
            return False
        
        # Check bounds
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < rs[i] + rs[j] - tol:
                    return False
        return True

    best_val = 1e10
    best_x = None
    
    # Try a few different initial aspect ratios and strategies
    aspect_ratios = [1.0, 0.8, 1.2, 0.6, 1.4, 0.5, 1.5, 0.7, 1.3]
    strategies = [0, 1, 2]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 10: # Leave 10s for polish
            break
        h_init = 2.0 - w_init
        
        for strat in strategies:
            if time.time() > deadline - 5:
                break
                
            xs, ys, rs = generate_grid_init(w_init, strat)
            x0_curr = np.concatenate(([w_init], xs, ys, rs))
            
            # First pass: coarse optimization
            try:
                res = minimize(objective, x0_curr, method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-10})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
            
    if best_x is None:
        # Fallback
        w0 = 1.0
        h0 = 1.0
        side = 5
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                cx = (i + 0.5) / side * w0
                cy = (j + 0.5) / side * h0
                r = min(cx, w0 - cx, cy, h0 - cy) * 0.9
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
        best_x = np.concatenate(([w0], np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])))
        best_val = objective(best_x)

    # Extract best solution
    w_best = best_x[0]
    h_best = 2.0 - w_best
    xs_best = best_x[1:N+1]
    ys_best = best_x[N+1:2*N+1]
    rs_best = best_x[2*N+1:3*N+1]
    
    # High-precision polish
    if time.time() < deadline - 5:
        x_polished, val_polished = polish_solution(best_x, w_best, max_iters=500)
        if val_polished < best_val:
            best_x = x_polished
            best_val = val_polished
            w_best = best_x[0]
            h_best = 2.0 - w_best
            xs_best = best_x[1:N+1]
            ys_best = best_x[N+1:2*N+1]
            rs_best = best_x[2*N+1:3*N+1]
            
    # Final validity check and repair
    circles = np.column_stack((xs_best, ys_best, rs_best))
    
    if not validate_solution(circles, w_best, h_best):
        # Attempt to fix using physics
        circles_fixed = resolve_collisions(circles, w_best, h_best, iters=500)
        
        # After fixing positions, ensure radii are valid (shrink if they stick out)
        for i in range(N):
            x, y, r = circles_fixed[i]
            max_r = min(x, w_best - x, y, h_best - y)
            if r > max_r:
                circles_fixed[i, 2] = max_r
            if circles_fixed[i, 2] < 0:
                circles_fixed[i, 2] = 0.001 # Small positive radius

        # Check again
        if validate_solution(circles_fixed, w_best, h_best):
            circles = circles_fixed
        else:
            # Aggressive shrink if still invalid
            for i in range(N):
                circles[i, 2] *= 0.9
            circles = resolve_collisions(circles, w_best, h_best, iters=100)
            for i in range(N):
                x, y, r = circles[i]
                max_r = min(x, w_best - x, y, h_best - y)
                if r > max_r:
                    circles[i, 2] = max_r

    return circles
