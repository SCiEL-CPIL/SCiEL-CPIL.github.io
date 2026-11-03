"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _total_radius(circles):
    return sum(c[2] for c in circles)

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are feasible: within bounds and non-overlapping."""
    for cx, cy, r in circles:
        if r < -tol or cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _make_grid(w, h, n=N):
    """Create a staggered grid of circles as initial guess."""
    circles = []
    for rows in range(3, 9):
        cols = int(np.ceil(n / rows))
        if rows * cols < n:
            continue
        r_est = min(w / cols, h / rows) * 0.42
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                cx = (j + 0.5) / cols * w
                cy = (i + 0.5) / rows * h
                if i % 2 == 1:
                    cx += 0.25 / cols * w
                r = min(r_est, min(cx, w - cx, cy, h - cy) * 0.85)
                r = max(r, 0.005)
                circles.append((cx, cy, r))
        if len(circles) == n:
            break
    while len(circles) < n:
        circles.append((w/2, h/2, 0.005))
    return circles[:n]

def _relax_circles(circles, w, h, iterations=50, lr=0.1):
    """Relax overlapping circles by pushing them apart."""
    circles = [list(c) for c in circles]
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist - 1e-9:
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = (min_dist - dist) * 0.5
                    nx, ny = dx/dist, dy/dist
                    circles[i][0] += nx * overlap
                    circles[i][1] += ny * overlap
                    circles[j][0] -= nx * overlap
                    circles[j][1] -= ny * overlap
                    moved = True
        # Clamp to bounds
        for i in range(N):
            cx, cy, r = circles[i]
            circles[i][0] = max(r, min(w - r, cx))
            circles[i][1] = max(r, min(h - r, cy))
        if not moved:
            break
    return [tuple(c) for c in circles]

def _slsqp_optimize(w, h, x0, maxiter=300, ftol=1e-10):
    """SLSQP with explicit constraints for circle packing."""
    def objective(x):
        return -np.sum(x[2::3])
    
    # Inequality constraints: for each pair, dist >= r_i + r_j
    # and for each circle, boundary constraints
    constraints = []
    
    # Boundary constraints: cx - r >= 0, cx + r <= w, cy - r >= 0, cy + r <= h
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
    
    # Pairwise non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2 - (x[i*3+2]+x[j*3+2])**2
            })
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, w - 0.001))
        bounds.append((0.001, h - 0.001))
        bounds.append((0.001, min(w, h) * 0.45))
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': maxiter, 'ftol': ftol})
    return res.x

def _polish_slsqp(w, h, x0, maxiter=500, ftol=1e-12):
    """High-precision polish with tighter tolerances."""
    def objective(x):
        return -np.sum(x[2::3])
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2 - (x[i*3+2]+x[j*3+2])**2
            })
    
    bounds = []
    for i in range(N):
        bounds.append((0.0005, w - 0.0005))
        bounds.append((0.0005, h - 0.0005))
        bounds.append((0.0005, min(w, h) * 0.45))
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': maxiter, 'ftol': ftol})
    return res.x

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    best_result = None
    best_score = 0.0
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 / (1 + aspect)
        
        initial_circles = _make_grid(w, h)
        x0 = np.array([v for c in initial_circles for v in c])
        
        for restart in range(4):
            if time.time() > deadline:
                break
            
            if restart == 0:
                x_init = x0.copy()
            else:
                x_init = x0 + np.random.normal(0, 0.015, size=x0.shape)
                # Clip to valid range
                for i in range(N):
                    x_init[i*3] = np.clip(x_init[i*3], 0.001, w - 0.001)
                    x_init[i*3+1] = np.clip(x_init[i*3+1], 0.001, h - 0.001)
                    x_init[i*3+2] = np.clip(x_init[i*3+2], 0.001, min(w, h) * 0.45)
            
            try:
                x_res = _slsqp_optimize(w, h, x_init, maxiter=250, ftol=1e-10)
                
                # Extract and validate
                circles = []
                for i in range(N):
                    cx = max(0.0005, min(x_res[i*3], w - 0.0005))
                    cy = max(0.0005, min(x_res[i*3+1], h - 0.0005))
                    r = max(0.0005, min(x_res[i*3+2], min(w, h) * 0.45))
                    circles.append((cx, cy, r))
                
                # Relax if needed
                if not _validate(circles, w, h):
                    circles = _relax_circles(circles, w, h)
                
                if _validate(circles, w, h):
                    score = _total_radius(circles)
                    if score > best_score:
                        best_score = score
                        best_result = circles
                        
            except Exception:
                pass
        
        # Polish the best result for this aspect if time allows
        if best_result is not None and time.time() < deadline - 20:
            try:
                x_polish = np.array([v for c in best_result for v in c])
                x_polished = _polish_slsqp(w, h, x_polish, maxiter=400, ftol=1e-12)
                
                circles_polished = []
                for i in range(N):
                    cx = max(0.0005, min(x_polished[i*3], w - 0.0005))
                    cy = max(0.0005, min(x_polished[i*3+1], h - 0.0005))
                    r = max(0.0005, min(x_polished[i*3+2], min(w, h) * 0.45))
                    circles_polished.append((cx, cy, r))
                
                if not _validate(circles_polished, w, h):
                    circles_polished = _relax_circles(circles_polished, w, h)
                
                if _validate(circles_polished, w, h):
                    score_polished = _total_radius(circles_polished)
                    if score_polished > best_score:
                        best_score = score_polished
                        best_result = circles_polished
                        
            except Exception:
                pass
    
    if best_result is not None:
        return np.array(best_result)
    
    # Fallback
    fallback = _make_grid(1.0, 1.0)
    return np.array(fallback)
# EVOLVE-BLOCK-END
