"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    This ensures the output is strictly valid.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < tol:
                        # If circles are on top of each other, push them apart randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Move apart by half the overlap each
                    overlap = (min_dist - dist) / 2.0
                    pos[i, 0] -= dx * overlap
                    pos[i, 1] -= dy * overlap
                    pos[j, 0] += dx * overlap
                    pos[j, 1] += dy * overlap
                    moved = True
        
        # Re-check boundaries after moving apart
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        if not moved:
            break

    return np.column_stack((pos, rad))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin under 120s limit
    
    best_val = -np.inf
    best_circles = None
    best_w = 1.0
    best_h = 1.0

    # Objective function for optimization
    # Variables: w, xs (N), ys (N), rs (N)
    # Total 1 + 3N variables
    def objective(v):
        w = v[0]
        if w <= 0 or w >= 2:
            return 1e10
        h = 2.0 - w
        
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        # Penalty for negative radii
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

    def generate_initial(w_init, strategy=0, seed=None):
        """Generate initial grid layout"""
        if seed is not None:
            np.random.seed(seed)
        h_init = 2.0 - w_init
        
        # Use a grid layout that fits N circles
        # 21 circles -> 5x5 grid is 25, so we use 5x5 and drop 4, or 3x7, etc.
        # Let's try a 5x5 grid centered
        cols = 5
        rows = 5
        xs = []
        ys = []
        rs = []
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                if strategy == 0:
                    # Standard grid
                    cx = (i + 0.5) / cols * w_init
                    cy = (j + 0.5) / rows * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.8
                elif strategy == 1:
                    # Offset grid (hexagonal-ish)
                    cx = (i + 0.5) / cols * w_init
                    if i % 2 == 1:
                        cy = (j + 0.75) / rows * h_init
                    else:
                        cy = (j + 0.25) / rows * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.75
                else:
                    # Randomized grid
                    cx = (i + np.random.uniform(0.3, 0.7)) / cols * w_init
                    cy = (j + np.random.uniform(0.3, 0.7)) / rows * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.7
                    
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
                count += 1
            
            if count >= N:
                break
        
        # If we didn't fill N, add random ones (shouldn't happen with 5x5=25)
        while len(xs) < N:
            xs.append(np.random.uniform(0.1, w_init-0.1))
            ys.append(np.random.uniform(0.1, h_init-0.1))
            rs.append(0.1)
            
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    # Try diverse aspect ratios
    # Perimeter 4 => w+h=2.
    # Square is 1x1.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5]
    
    for w_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        for strategy in range(3):
            if time.time() > deadline:
                break
                
            xs, ys, rs = generate_initial(w_init, strategy)
            x0 = np.concatenate(([w_init], xs, ys, rs))
            
            try:
                # Use L-BFGS-B for speed and stability
                res = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < best_val:
                    best_val = res.fun
                    w_best = res.x[0]
                    h_best = 2.0 - w_best
                    xs_best = res.x[1:N+1]
                    ys_best = res.x[N+1:2*N+1]
                    rs_best = res.x[2*N+1:3*N+1]
                    
                    # Clamp radii to be non-negative
                    rs_best = np.maximum(rs_best, 0)
                    
                    circles = np.column_stack((xs_best, ys_best, rs_best))
                    
                    # Relax to ensure validity
                    circles = _relax_circles(circles, w_best, h_best, iterations=500)
                    
                    # Check validity
                    if _validate(circles, w_best, h_best):
                        val = _total_radius(circles)
                        if val > best_val:
                            best_val = val
                            best_circles = circles
                            best_w = w_best
                            best_h = h_best
            except:
                pass

    if best_circles is None:
        # Fallback to simple grid
        w_init = 1.0
        xs, ys, rs = generate_initial(w_init, 0)
        best_w = w_init
        best_h = 2.0 - w_init
        best_circles = np.column_stack((xs, ys, rs))
        best_circles = _relax_circles(best_circles, best_w, best_h)

    return best_circles
