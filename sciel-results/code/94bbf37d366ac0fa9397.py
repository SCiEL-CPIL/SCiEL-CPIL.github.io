# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _objective(centers, sides, penalty_weight=1e8):
    """
    Objective: sum(sides) - penalty. Higher is better.
    """
    n = len(sides)
    penalty = 0.0
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < 0:
            penalty += (cx - s/2)**2
        if cx + s/2 > 1:
            penalty += (cx + s/2 - 1)**2
        if cy - s/2 < 0:
            penalty += (cy - s/2)**2
        if cy + s/2 > 1:
            penalty += (cy + s/2 - 1)**2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0
            if dx < t and dy < t:
                overlap_x = t - dx
                overlap_y = t - dy
                penalty += overlap_x * overlap_y
    
    return np.sum(sides) - penalty_weight * penalty

def _smooth_penalty(centers, sides, temperature=0.02):
    """
    Smooth penalty using soft-min approximation.
    Returns (penalty_value, gradients for centers and sides).
    """
    n = len(sides)
    penalty = 0.0
    grad_c = np.zeros_like(centers)
    grad_s = np.zeros_like(sides)
    
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Left boundary
        v = cx - s/2
        if v < 0:
            penalty += v * v
            grad_c[i, 0] += 2 * v
            grad_s[i] += -v
        # Right boundary
        v = 1 - (cx + s/2)
        if v < 0:
            penalty += v * v
            grad_c[i, 0] += 2 * v
            grad_s[i] += -v
        # Bottom boundary
        v = cy - s/2
        if v < 0:
            penalty += v * v
            grad_c[i, 1] += 2 * v
            grad_s[i] += -v
        # Top boundary
        v = 1 - (cy + s/2)
        if v < 0:
            penalty += v * v
            grad_c[i, 1] += 2 * v
            grad_s[i] += -v
    
    # Pairwise overlap penalties (smoothed)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            
            # Smooth approximation of max(0, t - |dx|)
            # Using soft-min: max(0, a) ≈ log(1 + exp(a/tau)) * tau
            ax = t - abs(dx)
            ay = t - abs(dy)
            
            if ax > -temperature and ay > -temperature:
                # Both dimensions have potential overlap
                soft_ax = temperature * np.log(1 + np.exp(ax / temperature)) if ax > 0 else 0
                soft_ay = temperature * np.log(1 + np.exp(ay / temperature)) if ay > 0 else 0
                
                # Approximate gradient
                if ax > 0:
                    sig_ax = 1.0 / (1.0 + np.exp(-ax / temperature))
                else:
                    sig_ax = 0.0
                if ay > 0:
                    sig_ay = 1.0 / (1.0 + np.exp(-ay / temperature))
                else:
                    sig_ay = 0.0
                
                # Gradient with respect to centers
                sign_dx = 1.0 if dx >= 0 else -1.0
                sign_dy = 1.0 if dy >= 0 else -1.0
                
                grad_c[i, 0] -= sign_dx * sig_ax * sig_ay
                grad_c[j, 0] += sign_dx * sig_ax * sig_ay
                grad_c[i, 1] -= sign_dy * sig_ax * sig_ay
                grad_c[j, 1] += sign_dy * sig_ax * sig_ay
                
                # Gradient with respect to sides
                grad_s[i] += sig_ax * sig_ay / 2.0
                grad_s[j] += sig_ax * sig_ay / 2.0
                
                penalty += soft_ax * soft_ay
    
    return penalty, grad_c, grad_s

def _gradient_ascent(centers, sides, deadline, lr_init=0.01, lr_decay=0.999, max_iter=2000):
    """
    Gradient ascent on smoothed objective: maximize sum(s) - penalty.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    lr = lr_init
    temperature = 0.02
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute penalty and gradients
        penalty, grad_c, grad_s = _smooth_penalty(best_c, best_s, temperature)
        
        # Gradient of objective = -gradient of penalty + 1 for sides
        obj_grad_c = -grad_c
        obj_grad_s = np.ones(n) - grad_s
        
        # Clip to valid ranges
        for i in range(n):
            s = best_s[i]
            cx = best_c[i, 0]
            cy = best_c[i, 1]
            
            # Adaptive learning rate based on how close to boundary
            margin_x = min(cx - s/2, 1 - cx - s/2)
            margin_y = min(cy - s/2, 1 - cy - s/2)
            min_margin = max(0, min(margin_x, margin_y))
            
            # Reduce step size near boundaries
            local_lr = lr * (1.0 + min_margin / 0.05)
            local_lr = min(local_lr, lr * 2.0)
            
            # Update
            new_cx = cx + local_lr * obj_grad_c[i, 0]
            new_cy = cy + local_lr * obj_grad_c[i, 1]
            new_s = s + local_lr * obj_grad_s[i]
            
            # Clip
            new_s = np.clip(new_s, 0.01, 0.5)
            new_cx = np.clip(new_cx, new_s/2, 1 - new_s/2)
            new_cy = np.clip(new_cy, new_s/2, 1 - new_s/2)
            
            best_c[i, 0] = new_cx
            best_c[i, 1] = new_cy
            best_s[i] = new_s
        
        # Check if valid and update best
        if _is_valid(best_c, best_s):
            score = np.sum(best_s)
            if score > best_score:
                best_score = score
        
        # Decay learning rate
        lr *= lr_decay
        # Anneal temperature
        if it % 100 == 0 and it > 0:
            temperature *= 0.8
    
    return best_c, best_s

def _scipy_optimize(centers, sides, deadline):
    """
    Uses scipy.optimize.minimize to refine the configuration.
    """
    n = len(sides)
    # Flatten to vector: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.concatenate([centers.flatten(), sides])
    
    def fun(x):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop
        
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Clip to valid ranges to help optimizer
        s_clipped = np.clip(s, 0.01, 0.5)
        
        c_clipped = np.zeros_like(c)
        for i in range(n):
            c_clipped[i, 0] = np.clip(c[i, 0], s_clipped[i]/2, 1 - s_clipped[i]/2)
            c_clipped[i, 1] = np.clip(c[i, 1], s_clipped[i]/2, 1 - s_clipped[i]/2)
            
        # We want to maximize sum(s) - penalty
        # So minimize - (sum(s) - penalty)
        obj = _objective(c_clipped, s_clipped)
        return -obj

    # Use Nelder-Mead, which is derivative-free and robust
    try:
        res = minimize(fun, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    except Exception:
        return centers, sides

    x_opt = res.x
    c_opt = x_opt[:2*n].reshape(n, 2)
    s_opt = x_opt[2*n:]
    
    # Final clipping
    s_opt = np.clip(s_opt, 0.01, 0.5)
    for i in range(n):
        c_opt[i, 0] = np.clip(c_opt[i, 0], s_opt[i]/2, 1 - s_opt[i]/2)
        c_opt[i, 1] = np.clip(c_opt[i, 1], s_opt[i]/2, 1 - s_opt[i]/2)
        
    return c_opt, s_opt

def _local_search(centers, sides, max_iters=3000, deadline=None, rng_seed=42):
    """
    More aggressive local search with multiple strategies.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(rng_seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Strategy 1: Perturb a single square
        i = rng.randint(n)
        for attempt in range(8):
            # Try to grow
            grow = rng.random() < 0.7
            if grow:
                s_new = best_s[i] * (1 + rng.uniform(0.01, 0.08))
            else:
                s_new = best_s[i] * (1 + rng.uniform(-0.03, 0.03))
            s_new = max(0.01, min(0.5, s_new))
            
            cx_new = best_c[i, 0] + rng.uniform(-0.05, 0.05)
            cy_new = best_c[i, 1] + rng.uniform(-0.05, 0.05)
            
            cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
            cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
            
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            c_new[i] = [cx_new, cy_new]
            s_new_arr[i] = s_new
            
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
                    improved = True
                    break
        
        # Strategy 2: Perturb two squares
        if not improved and rng.random() < 0.4:
            i1 = rng.randint(n)
            i2 = rng.randint(n)
            if i1 != i2:
                for attempt in range(4):
                    c_new = best_c.copy()
                    s_new_arr = best_s.copy()
                    for idx in [i1, i2]:
                        s_new = s_new_arr[idx] * (1 + rng.uniform(-0.02, 0.05))
                        s_new = max(0.01, min(0.5, s_new))
                        cx_new = c_new[idx, 0] + rng.uniform(-0.03, 0.03)
                        cy_new = c_new[idx, 1] + rng.uniform(-0.03, 0.03)
                        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                        c_new[idx] = [cx_new, cy_new]
                        s_new_arr[idx] = s_new
                    
                    if _is_valid(c_new, s_new_arr):
                        score = np.sum(s_new_arr)
                        if score > best_score + 1e-12:
                            best_score = score
                            best_c = c_new
                            best_s = s_new_arr
                            break
        
        # Strategy 3: Global random restart (rare)
        if not improved and rng.random() < 0.02:
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            for idx in range(n):
                s_new = s_new_arr[idx] * (1 + rng.uniform(-0.1, 0.1))
                s_new = max(0.01, min(0.5, s_new))
                cx_new = rng.uniform(s_new/2, 1 - s_new/2)
                cy_new = rng.uniform(s_new/2, 1 - s_new/2)
                c_new[idx] = [cx_new, cy_new]
                s_new_arr[idx] = s_new
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
    
    return best_c, best_s

def _coordinate_descent(centers, sides, max_iters=500, deadline=None):
    """
    Coordinate descent: for each square, optimize its position and size.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            best_local_score = np.sum(best_s)
            best_local_c = best_c.copy()
            best_local_s = best_s.copy()
            
            # Try to optimize size first
            for s_try in np.linspace(0.05, 0.5, 20):
                if s_try <= best_s[i] * 0.8 or s_try >= best_s[i] * 1.2:
                    continue
                # Try to find a valid position
                for _ in range(30):
                    cx = np.random.uniform(s_try/2, 1 - s_try/2)
                    cy = np.random.uniform(s_try/2, 1 - s_try/2)
                    
                    c_try = best_local_c.copy()
                    s_try_arr = best_local_s.copy()
                    c_try[i] = [cx, cy]
                    s_try_arr[i] = s_try
                    
                    if _is_valid(c_try, s_try_arr):
                        score = np.sum(s_try_arr)
                        if score > best_local_score:
                            best_local_score = score
                            best_local_c = c_try
                            best_local_s = s_try_arr
            
            # Now optimize position with fixed size
            s_fixed = best_local_s[i]
            for _ in range(20):
                cx = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                cy = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                
                c_try = best_local_c.copy()
                c_try[i] = [cx, cy]
                
                if _is_valid(c_try, best_local_s):
                    score = np.sum(best_local_s)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_c = c_try
            
            best_c = best_local_c
            best_s = best_local_s
            best_score = best_local_score
    
    return best_c, best_s

def _simulated_annealing(centers, sides, deadline, T_init=0.05, T_min=0.001, cooling=0.999, max_iter=5000):
    """
    Simulated annealing to escape local minima.
    """
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = np.sum(sides)
    
    current_c = centers.copy()
    current_s = sides.copy()
    current_score = np.sum(sides)
    
    T = T_init
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        c_new = current_c.copy()
        s_new = current_s.copy()
        
        # Perturb 1-2 squares
        num_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(n, size=num_perturb, replace=False)
        
        for idx in indices:
            # Perturb size
            s_new[idx] *= (1 + rng.uniform(-0.05, 0.05))
            s_new[idx] = np.clip(s_new[idx], 0.01, 0.5)
            
            # Perturb position
            c_new[idx, 0] += rng.uniform(-0.03, 0.03)
            c_new[idx, 1] += rng.uniform(-0.03, 0.03)
            
            # Clip to valid range
            c_new[idx, 0] = np.clip(c_new[idx, 0], s_new[idx]/2, 1 - s_new[idx]/2)
            c_new[idx, 1] = np.clip(c_new[idx, 1], s_new[idx]/2, 1 - s_new[idx]/2)
        
        # Check validity
        if not _is_valid(c_new, s_new):
            # Reject invalid moves
            T *= cooling
            continue
        
        new_score = np.sum(s_new)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_c = c_new
            current_s = s_new
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_c = c_new.copy()
                best_s = s_new.copy()
        
        # Cool down
        T *= cooling
    
    return best_c, best_s

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Generate diverse initial configurations
    configs = []
    
    # 2x5 grid
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    configs.append((c1, s1))
    
    # 1x10 strip
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        c2[i] = [(i + 0.5) * 0.1, 0.5]
        s2[i] = 0.1
    configs.append((c2, s2))
    
    # 5x2 grid
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i % 2
        c = i // 2
        c3[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s3[i] = 0.2
    configs.append((c3, s3))
    
    # 3x3 grid + 1
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        c4[i] = [(c + 0.5) / 3.0, (r + 0.5) / 3.0]
        s4[i] = 1.0/3.0
    c4[9] = [0.5, 0.5]
    s4[9] = 0.1
    configs.append((c4, s4))
    
    # Random initializations with greedy placement
    rng = np.random.RandomState(123)
    for trial in range(40):
        if time.time() > deadline:
            break
        
        s_rand = rng.uniform(0.08, 0.35, n)
        c_rand = np.zeros((n, 2))
        placed = np.zeros(n, dtype=bool)
        
        order = np.argsort(-s_rand)
        
        for idx in order:
            s = s_rand[idx]
            found = False
            for _ in range(300):
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                c_rand[idx] = [cx, cy]
                
                ok = True
                for j in range(n):
                    if placed[j] and j != idx:
                        dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                        dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                        t = (s_rand[idx] + s_rand[j]) / 2.0
                        if dx < t and dy < t:
                            ok = False
                            break
                if ok:
                    placed[idx] = True
                    found = True
                    break
            
            if not found:
                s_rand[idx] = max(0.05, s_rand[idx] * 0.8)
        
        if _is_valid(c_rand, s_rand):
            configs.append((c_rand, s_rand))
    
    # For each config, run optimization
    for c_init, s_init in configs:
        if time.time() > deadline:
            break
        
        update_best(c_init, s_init)
        
        # 1. Run SciPy Optimizer (Stronger)
        if time.time() < deadline - 10:
            c_scipy, s_scipy = _scipy_optimize(c_init, s_init, deadline)
            update_best(c_scipy, s_scipy)
            
            # 2. Run Gradient Ascent on the result of SciPy
            if time.time() < deadline - 5:
                c_ga, s_ga = _gradient_ascent(c_scipy, s_scipy, deadline, lr_init=0.005, lr_decay=0.9995, max_iter=1500)
                update_best(c_ga, s_ga)
            
            # 3. Run Simulated Annealing on the best result
            if time.time() < deadline - 5:
                best_for_sa = c_scipy if np.sum(s_scipy) >= np.sum(s_ga) else c_ga
                best_s_for_sa = s_scipy if np.sum(s_scipy) >= np.sum(s_ga) else s_ga
                c_sa, s_sa = _simulated_annealing(best_for_sa, best_s_for_sa, deadline, T_init=0.03, T_min=0.0005, cooling=0.9995, max_iter=3000)
                update_best(c_sa, s_sa)
            
            # 4. Run Local Search on the best result (Polish)
            if time.time() < deadline - 5:
                best_for_ls = c_sa if np.sum(s_sa) >= np.sum(s_ga) else c_ga
                best_s_for_ls = s_sa if np.sum(s_sa) >= np.sum(s_ga) else s_ga
                c_ls, s_ls = _local_search(best_for_ls, best_s_for_ls, max_iters=200, deadline=deadline, rng_seed=42)
                update_best(c_ls, s_ls)
        
    # Final polish: try to squeeze more out of the best configuration
    if best_centers is not None and time.time() < deadline - 20:
        c_final, s_final = _scipy_optimize(best_centers, best_sides, deadline)
        update_best(c_final, s_final)
        
        # Simulated Annealing polish
        if time.time() < deadline - 10:
            c_sa_final, s_sa_final = _simulated_annealing(c_final, s_final, deadline, T_init=0.01, T_min=0.0001, cooling=0.9998, max_iter=2000)
            update_best(c_sa_final, s_sa_final)
        
        # Gradient ascent polish
        if time.time() < deadline - 10:
            c_ga_final, s_ga_final = _gradient_ascent(c_final, s_final, deadline, lr_init=0.002, lr_decay=0.9998, max_iter=1000)
            update_best(c_ga_final, s_ga_final)
        
        # Local search polish
        if time.time() < deadline - 5:
            best_for_final_ls = c_sa_final if np.sum(s_sa_final) >= np.sum(s_ga_final) else c_ga_final
            best_s_for_final_ls = s_sa_final if np.sum(s_sa_final) >= np.sum(s_ga_final) else s_ga_final
            c_final2, s_final2 = _local_search(best_for_final_ls, best_s_for_final_ls, max_iters=500, deadline=deadline, rng_seed=777)
            update_best(c_final2, s_final2)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
