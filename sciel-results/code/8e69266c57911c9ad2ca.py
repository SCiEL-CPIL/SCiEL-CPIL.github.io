# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog, differential_evolution as de
from itertools import combinations

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
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

def _penalty_obj(x, n=10, lam=1e8):
    """
    x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]
    Returns: -sum(s_i) + penalties
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    obj = -np.sum(sides)
    
    # Boundary violations
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < 0:
            obj += lam * (cx - s/2)**2
        if cx + s/2 > 1:
            obj += lam * (cx + s/2 - 1)**2
        if cy - s/2 < 0:
            obj += lam * (cy - s/2)**2
        if cy + s/2 > 1:
            obj += lam * (cy + s/2 - 1)**2
    
    # Overlap violations
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            if abs(dx) < t and abs(dy) < t:
                ox = t - abs(dx)
                oy = t - abs(dy)
                obj += lam * (ox * oy)
    
    return obj

def _penalty_grad(x, n=10, lam=1e8):
    """Gradient of penalty objective."""
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    grad = np.zeros(3*n)
    
    # Gradient of -sum(sides)
    grad[2*n:] = -1.0
    
    # Boundary gradients
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # x boundaries
        if cx - s/2 < 0:
            v = cx - s/2
            grad[i] += lam * 2 * v * 1.0
            grad[2*n+i] += lam * 2 * v * (-0.5)
        if cx + s/2 > 1:
            v = cx + s/2 - 1
            grad[i] += lam * 2 * v * 1.0
            grad[2*n+i] += lam * 2 * v * 0.5
        # y boundaries
        if cy - s/2 < 0:
            v = cy - s/2
            grad[n+i] += lam * 2 * v * 1.0
            grad[2*n+i] += lam * 2 * v * (-0.5)
        if cy + s/2 > 1:
            v = cy + s/2 - 1
            grad[n+i] += lam * 2 * v * 1.0
            grad[2*n+i] += lam * 2 * v * 0.5
    
    # Overlap gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            if abs(dx) < t and abs(dy) < t:
                ox = t - abs(dx)
                oy = t - abs(dy)
                # d/dx_i of ox*oy
                if dx >= 0:
                    grad[i] += lam * (-oy)
                    grad[j] += lam * (oy)
                else:
                    grad[i] += lam * (oy)
                    grad[j] += lam * (-oy)
                if dy >= 0:
                    grad[n+i] += lam * (-ox)
                    grad[n+j] += lam * (ox)
                else:
                    grad[n+i] += lam * (ox)
                    grad[n+j] += lam * (-ox)
                # d/ds_i of ox*oy
                grad[2*n+i] += lam * (0.5 * oy)
                grad[2*n+j] += lam * (0.5 * ox)
    
    return grad

def _local_search(centers, sides, max_iters=2000, deadline=None, step_scale=0.02):
    """
    Local search with adaptive step sizes.
    """
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = np.sum(sides)
    
    rng = np.random.RandomState(123)
    step = step_scale
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(n)
        
        # Try perturbing position and/or size
        dcx = rng.uniform(-step, step)
        dcy = rng.uniform(-step, step)
        ds = rng.uniform(-step, step) * best_s[i]
        
        cx_new = best_c[i, 0] + dcx
        cy_new = best_c[i, 1] + dcy
        s_new = best_s[i] + ds
        s_new = max(0.01, min(0.5, s_new))
        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
        
        c_new = best_c.copy()
        s_new_arr = best_s.copy()
        c_new[i] = [cx_new, cy_new]
        s_new_arr[i] = s_new
        
        if _is_valid(c_new, s_new_arr):
            score = np.sum(s_new_arr)
            if score > best_score:
                best_score = score
                best_c = c_new
                best_s = s_new_arr
                step = min(step * 1.1, 0.1)  # increase step slightly
            else:
                step = max(step * 0.95, 0.001)  # decrease step
        else:
            step = max(step * 0.9, 0.001)
    
    return best_c, best_s

def _gradient_ascent(centers, sides, deadline, max_iters=500):
    """
    Gradient-based local optimization using the penalty method.
    """
    n = len(sides)
    x = np.concatenate([centers.flatten(), sides])
    
    best_score = np.sum(sides)
    best_x = x.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        obj = _penalty_obj(x, n)
        grad = _penalty_grad(x, n)
        
        # Take gradient ascent step (negative gradient since we minimize)
        x_new = x - lr * grad
        
        # Clamp
        for i in range(n):
            s = x_new[2*n+i]
            s = max(0.01, min(0.5, s))
            x_new[2*n+i] = s
            x_new[i] = max(s/2, min(1-s/2, x_new[i]))
            x_new[n+i] = max(s/2, min(1-s/2, x_new[n+i]))
        
        c_new = x_new[:2*n].reshape(n, 2)
        s_new = x_new[2*n:]
        
        if _is_valid(c_new, s_new):
            score = np.sum(s_new)
            if score > best_score:
                best_score = score
                best_x = x_new.copy()
                lr = min(lr * 1.05, 0.05)
            else:
                lr = max(lr * 0.9, 0.001)
        else:
            # Roll back and reduce learning rate
            lr = max(lr * 0.8, 0.0005)
            x_new = x  # keep old
        x = x_new
    
    centers_out = best_x[:2*n].reshape(n, 2)
    sides_out = best_x[2*n:]
    return centers_out, sides_out

def _build_lp_with_ordering(centers_init, sides_init, n=10, M=2.0):
    """
    Build and solve LP with fixed orientation ordering based on initial positions.
    """
    num_pairs = n * (n - 1) // 2
    total_vars = 2 * n + n + 4 * num_pairs
    
    c_obj = np.zeros(total_vars)
    for i in range(n):
        c_obj[2 * n + i] = -1.0
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(total_vars)
        row[i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        row = np.zeros(total_vars)
        row[n + i] = -1.0
        row[2 * n + i] = -0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(total_vars)
        row[n + i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    pair_counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            z_base = 3 * n + 4 * pair_counter
            
            # Determine initial orientation
            dx = centers_init[i, 0] - centers_init[j, 0]
            dy = centers_init[i, 1] - centers_init[j, 1]
            
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 0] = M
            A_ub.append(row)
            b_ub.append(M)
            
            row = np.zeros(total_vars)
            row[i] = -1.0
            row[j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 1] = M
            A_ub.append(row)
            b_ub.append(M)
            
            row = np.zeros(total_vars)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 2] = M
            A_ub.append(row)
            b_ub.append(M)
            
            row = np.zeros(total_vars)
            row[n + i] = -1.0
            row[n + j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            row[z_base + 3] = M
            A_ub.append(row)
            b_ub.append(M)
            
            pair_counter += 1
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0.0, 1.0)] * (2 * n)
    bounds += [(0.01, 0.5)] * n
    bounds += [(0.0, 1.0)] * (4 * num_pairs)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        x = res.x
        centers_lp = x[:2 * n].reshape(n, 2)
        sides_lp = x[2 * n:3 * n]
        return centers_lp, sides_lp, True
    
    return None, None, False

def _fixed_orientation_lp(centers, sides, n=10):
    """
    Given fixed side lengths, find optimal centers using LP with orientations
    determined by current relative positions.
    """
    num_pairs = n * (n - 1) // 2
    total_vars = 2 * n
    
    c_obj = np.zeros(total_vars)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[n+i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[n+i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            sep = (sides[i] + sides[j]) / 2.0
            
            if abs(dx) > abs(dy):
                # Horizontal separation
                if dx >= 0:
                    # i is to the right of j
                    row = np.zeros(total_vars)
                    row[i] = 1.0
                    row[j] = -1.0
                    A_ub.append(row)
                    b_ub.append(sep)
                else:
                    # i is to the left of j
                    row = np.zeros(total_vars)
                    row[i] = -1.0
                    row[j] = 1.0
                    A_ub.append(row)
                    b_ub.append(sep)
            else:
                # Vertical separation
                if dy >= 0:
                    # i is above j
                    row = np.zeros(total_vars)
                    row[n+i] = 1.0
                    row[n+j] = -1.0
                    A_ub.append(row)
                    b_ub.append(sep)
                else:
                    # i is below j
                    row = np.zeros(total_vars)
                    row[n+i] = -1.0
                    row[n+j] = 1.0
                    A_ub.append(row)
                    b_ub.append(sep)
    
    if not A_ub:
        return centers.copy(), True
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (2*n), method='highs')
    if res.status == 0:
        centers_out = res.x.reshape(n, 2)
        return centers_out, True
    return None, False

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
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
    
    # ---- Initial seed: grid-based packing ----
    # Try a 2x5 grid of equal squares
    c_seed = np.zeros((n, 2))
    s_seed = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c_seed[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s_seed[i] = 0.2
    update_best(c_seed, s_seed)
    
    # Try 5x2 grid
    c_seed2 = np.zeros((n, 2))
    s_seed2 = np.zeros(n)
    for i in range(n):
        r = i // 2
        c = i % 2
        c_seed2[i] = [(c + 0.5) * 0.5, (r + 0.5) * 0.2]
        s_seed2[i] = 0.2
    update_best(c_seed2, s_seed2)
    
    # Try mixed: 2 large + 8 small
    c_seed3 = np.zeros((n, 2))
    s_seed3 = np.zeros(n)
    c_seed3[0] = [0.25, 0.25]
    s_seed3[0] = 0.5
    c_seed3[1] = [0.75, 0.75]
    s_seed3[1] = 0.5
    small_pos = [
        [0.75, 0.25], [0.25, 0.75], [0.5, 0.5],
        [0.125, 0.5], [0.875, 0.5], [0.5, 0.125],
        [0.5, 0.875], [0.125, 0.125]
    ]
    for i in range(8):
        c_seed3[i+2] = small_pos[i]
        s_seed3[i+2] = 0.25
    update_best(c_seed3, s_seed3)
    
    # Try 4x4 grid (16 squares, take 10)
    c_seed4 = np.zeros((n, 2))
    s_seed4 = np.zeros(n)
    for i in range(n):
        r = i // 4
        c = i % 4
        c_seed4[i] = [(c + 0.5) * 0.25, (r + 0.5) * 0.25]
        s_seed4[i] = 0.25
    update_best(c_seed4, s_seed4)
    
    # ---- Local search from best seed ----
    if best_centers is not None:
        c_ls, s_ls = _local_search(best_centers, best_sides, max_iters=3000, deadline=deadline)
        update_best(c_ls, s_ls)
    
    # ---- Gradient ascent from best ----
    if best_centers is not None and time.time() < deadline:
        c_grad, s_grad = _gradient_ascent(best_centers, best_sides, deadline, max_iters=300)
        update_best(c_grad, s_grad)
    
    # ---- LP with ordering from best ----
    if best_centers is not None and time.time() < deadline:
        c_lp, s_lp, ok = _build_lp_with_ordering(best_centers, best_sides, n, M=2.0)
        if ok:
            update_best(c_lp, s_lp)
            # Local search from LP result
            c_ls2, s_ls2 = _local_search(c_lp, s_lp, max_iters=2000, deadline=deadline)
            update_best(c_ls2, s_ls2)
            # Gradient from LP result
            if time.time() < deadline:
                c_grad2, s_grad2 = _gradient_ascent(c_ls2, s_ls2, deadline, max_iters=200)
                update_best(c_grad2, s_grad2)
    
    # ---- Differential Evolution ----
    if time.time() < deadline:
        bounds = [(0.01, 0.5)] * n  # sides
        # We'll use DE on a reduced representation: just sides, then LP for centers
        # Actually, let's do full DE on all variables
        
        def de_obj(x):
            centers = x[:2*n].reshape(n, 2)
            sides = x[2*n:]
            if not _is_valid(centers, sides):
                return 1e6 - np.sum(sides)
            return -np.sum(sides)
        
        de_bounds = []
        for i in range(n):
            de_bounds.append((0.01, 1.0))  # cx
        for i in range(n):
            de_bounds.append((0.01, 1.0))  # cy
        for i in range(n):
            de_bounds.append((0.01, 0.5))  # sides
        
        try:
            res_de = de(de_obj, de_bounds, maxiter=100, popsize=20, 
                       tol=1e-6, seed=42, workers=-1, 
                       polish=False, init='sobol')
            if res_de.fun < 1e5:  # valid
                x_de = res_de.x
                c_de = x_de[:2*n].reshape(n, 2)
                s_de = x_de[2*n:]
                # Clamp to valid range
                for i in range(n):
                    s_de[i] = max(0.01, min(0.5, s_de[i]))
                    c_de[i, 0] = max(s_de[i]/2, min(1-s_de[i]/2, c_de[i, 0]))
                    c_de[i, 1] = max(s_de[i]/2, min(1-s_de[i]/2, c_de[i, 1]))
                update_best(c_de, s_de)
                
                # Local search from DE result
                if time.time() < deadline:
                    c_ls3, s_ls3 = _local_search(c_de, s_de, max_iters=2000, deadline=deadline)
                    update_best(c_ls3, s_ls3)
                    if time.time() < deadline:
                        c_grad3, s_grad3 = _gradient_ascent(c_ls3, s_ls3, deadline, max_iters=200)
                        update_best(c_grad3, s_grad3)
        except Exception:
            pass
    
    # ---- Iterative LP refinement ----
    if best_centers is not None and time.time() < deadline:
        c_cur = best_centers.copy()
        s_cur = best_sides.copy()
        for iteration in range(5):
            if time.time() > deadline:
                break
            c_fixed, ok = _fixed_orientation_lp(c_cur, s_cur, n)
            if ok and _is_valid(c_fixed, s_cur):
                c_cur = c_fixed
                # Try to increase sides slightly
                s_try = s_cur * 1.01
                s_try = np.clip(s_try, 0.01, 0.5)
                c_try, ok2 = _fixed_orientation_lp(c_cur, s_try, n)
                if ok2 and _is_valid(c_try, s_try):
                    update_best(c_try, s_try)
                    c_cur = c_try
                    s_cur = s_try.copy()
                else:
                    # Try individual side increases
                    for i in range(n):
                        if time.time() > deadline:
                            break
                        s_ind = s_cur.copy()
                        s_ind[i] = min(0.5, s_cur[i] * 1.02)
                        c_ind, ok3 = _fixed_orientation_lp(c_cur, s_ind, n)
                        if ok3 and _is_valid(c_ind, s_ind):
                            update_best(c_ind, s_ind)
                            c_cur = c_ind
                            s_cur = s_ind.copy()
    
    # ---- Final local search polish ----
    if best_centers is not None and time.time() < deadline:
        c_final, s_final = _local_search(best_centers, best_sides, max_iters=3000, deadline=deadline)
        update_best(c_final, s_final)
    
    # ---- Return best found ----
    if best_centers is None:
        # Fallback: 2x5 grid
        best_centers = np.zeros((n, 2))
        best_sides = np.zeros(n)
        for i in range(n):
            r = i // 5
            c = i % 5
            best_centers[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
            best_sides[i] = 0.2
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
