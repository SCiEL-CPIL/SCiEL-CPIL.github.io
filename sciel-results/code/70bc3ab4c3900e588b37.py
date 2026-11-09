import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 620
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(centers, sides, tol=1e-7):
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
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True
    
    def score(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if centers is None or sides is None:
            return
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return
        if np.any(sides <= 0):
            return
        s = score(centers, sides)
        if s > best_score:
            best_score = s
            best_centers = centers.copy()
            best_sides = sides.copy()
            
    # 1. Construct the known 3.0 solution
    c_known = np.zeros((n, 2))
    s_known = np.zeros(n)
    # 2 large
    c_known[0] = [0.25, 0.75]
    s_known[0] = 0.5
    c_known[1] = [0.75, 0.75]
    s_known[1] = 0.5
    # 8 small
    for r in range(2):
        for c in range(4):
            idx = 2 + r*4 + c
            c_known[idx, 0] = 0.125 + c*0.25
            c_known[idx, 1] = 0.125 + r*0.25
            s_known[idx] = 0.25
    update_best(c_known, s_known)
    
    # 2. Define objective for optimization using smooth soft-constraint loss
    def soft_max(x, beta=100.0):
        """Smooth maximum of a vector using log-sum-exp approximation."""
        x_max = np.max(x)
        if x_max > 1e10:
            return x_max
        shifted = x - x_max
        return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    def objective(params, beta=50.0):
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        
        # Sides must be positive
        if np.any(sides <= 0):
            return 1e6
            
        obj = -np.sum(sides)
        
        # Boundary penalty (smooth)
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left
            v = max(0, -(cx - s/2))
            obj += 100 * v*v
            # Right
            v = max(0, (cx + s/2) - 1)
            obj += 100 * v*v
            # Bottom
            v = max(0, -(cy - s/2))
            obj += 100 * v*v
            # Top
            v = max(0, (cy + s/2) - 1)
            obj += 100 * v*v
            
        # Overlap penalty using smooth maximum
        tol = 1e-7
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = max(0, t - dx)
                oy = max(0, t - dy)
                if ox > 0 and oy > 0:
                    # Use smooth product via soft_max of overlap components
                    obj += 100 * ox * oy
                    
        return obj
    
    # 3. Run optimization from multiple starts
    # Start 1: Known solution
    starts = [np.concatenate([c_known.flatten(), s_known])]
    
    # Start 2: Random valid-ish packing
    c_rand = np.zeros((n, 2))
    s_rand = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c_rand[i, 0] = (c + 0.5) * 0.2
        c_rand[i, 1] = (r + 0.5) * 0.5
        s_rand[i] = 0.2
    starts.append(np.concatenate([c_rand.flatten(), s_rand]))
    
    # Start 3: Perturbed known solution
    c_pert = c_known + np.random.normal(0, 0.01, (n, 2))
    s_pert = s_known + np.random.normal(0, 0.01, n)
    s_pert = np.clip(s_pert, 0.1, 0.6)
    starts.append(np.concatenate([c_pert.flatten(), s_pert]))
    
    # Start 4: Different perturbation
    c_pert2 = c_known + np.random.normal(0, 0.02, (n, 2))
    s_pert2 = s_known + np.random.normal(0, 0.02, n)
    s_pert2 = np.clip(s_pert2, 0.1, 0.6)
    starts.append(np.concatenate([c_pert2.flatten(), s_pert2]))
    
    for init in starts:
        if elapsed() > 580:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14})
            if res.fun < 0:
                c_opt = res.x[:2*n].reshape(n, 2)
                s_opt = np.maximum(res.x[2*n:], 1e-6)
                update_best(c_opt, s_opt)
        except Exception:
            pass
            
    # 4. Polish pass: high-precision refinement on best solution
    if best_centers is not None and elapsed() < 590:
        try:
            init_polish = np.concatenate([best_centers.flatten(), best_sides])
            res_polish = minimize(objective, init_polish, method='L-BFGS-B',
                                   options={'maxiter': 5000, 'ftol': 1e-16})
            if res_polish.fun < 0:
                c_polish = res_polish.x[:2*n].reshape(n, 2)
                s_polish = np.maximum(res_polish.x[2*n:], 1e-6)
                update_best(c_polish, s_polish)
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = c_known
        best_sides = s_known
        
    return best_centers, best_sides
