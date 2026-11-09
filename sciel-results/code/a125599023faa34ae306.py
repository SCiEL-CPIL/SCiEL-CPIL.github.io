import numpy as np
import time
from scipy.optimize import minimize

_TOL = 1e-7

def _objective(x, beta):
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Penalty for negative sides
    side_pen = beta * np.sum(np.maximum(0, -sides)**2)
    
    # Penalty for boundary violations
    bound_pen = 0.0
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        v_left = max(0, s/2 - cx)
        v_right = max(0, cx + s/2 - 1)
        v_bottom = max(0, s/2 - cy)
        v_top = max(0, cy + s/2 - 1)
        bound_pen += beta * (v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
        
    # Penalty for overlaps
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # Overlap area approximation
            ox = max(0, t - dx)
            oy = max(0, t - dy)
            overlap_pen += beta * (ox * oy) ** 2
            
    return -np.sum(sides) + bound_pen + overlap_pen + side_pen

def _is_feasible(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _make_init_30():
    # 2x0.5, 4x0.25, 4x0.25
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # Row 1: 2 squares of 0.5
    centers[0] = [0.25, 0.25]; sides[0] = 0.5
    centers[1] = [0.75, 0.25]; sides[1] = 0.5
    # Row 2: 4 squares of 0.25
    centers[2] = [0.125, 0.625]; sides[2] = 0.25
    centers[3] = [0.375, 0.625]; sides[3] = 0.25
    centers[4] = [0.625, 0.625]; sides[4] = 0.25
    centers[5] = [0.875, 0.625]; sides[5] = 0.25
    # Row 3: 4 squares of 0.25
    centers[6] = [0.125, 0.875]; sides[6] = 0.25
    centers[7] = [0.375, 0.875]; sides[7] = 0.25
    centers[8] = [0.625, 0.875]; sides[8] = 0.25
    centers[9] = [0.875, 0.875]; sides[9] = 0.25
    return centers, sides

def _make_init_30_alt():
    # 4x0.25, 3x0.333, 3x0.333
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    for i in range(4):
        centers[i] = [0.125 + 0.25*i, 0.125]
        sides[i] = 0.25
    for i in range(3):
        centers[4+i] = [1/6 + i/3, 0.25 + 1/6]
        sides[4+i] = 1/3
    for i in range(3):
        centers[7+i] = [1/6 + i/3, 0.25 + 1/3 + 1/6]
        sides[7+i] = 1/3
    return centers, sides

def squares_in_square10() -> tuple:
    deadline = time.time() + 580
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    inits = [
        _make_init_30(),
        _make_init_30_alt(),
        # Add a random init
        (np.random.rand(10, 2), np.random.rand(10) * 0.3 + 0.1)
    ]
    
    for centers0, sides0 in inits:
        if time.time() > deadline: break
        x0 = np.concatenate([centers0.ravel(), sides0])
        
        # Anneal beta
        for beta in [1.0, 10.0, 100.0, 1000.0]:
            if time.time() > deadline: break
            try:
                res = minimize(_objective, x0, args=(beta,), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
                x_opt = res.x
                centers_opt = x_opt[:20].reshape(10, 2)
                sides_opt = x_opt[20:]
                
                centers_opt = np.clip(centers_opt, 0, 1)
                sides_opt = np.clip(sides_opt, 1e-6, 1.0)
                
                if _is_feasible(centers_opt, sides_opt):
                    score = np.sum(sides_opt)
                    if score > best_score:
                        best_score = score
                        best_centers = centers_opt.copy()
                        best_sides = sides_opt.copy()
                else:
                    # Try shrinking
                    for scale in [0.99, 0.95, 0.9]:
                        sides_try = sides_opt * scale
                        if _is_feasible(centers_opt, sides_try):
                            score = np.sum(sides_try)
                            if score > best_score:
                                best_score = score
                                best_centers = centers_opt.copy()
                                best_sides = sides_try.copy()
                            break
                x0 = x_opt
            except:
                pass

    if best_centers is None:
        # Fallback
        best_centers = np.array([[0.5, 0.5]] * 10)
        best_sides = np.full(10, 0.1)
        
    return best_centers.astype(np.float64), best_sides.astype(np.float64)
