import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    # Variables: x1, y1, s1, ..., x10, y10, s10
    # Total 30 variables
    
    def objective(v):
        # v is shape (30,)
        x = v[0:30:3]
        y = v[1:30:3]
        s = v[2:30:3]
        
        # 1. Boundary penalties
        # x - s/2 >= 0, x + s/2 <= 1
        # y - s/2 >= 0, y + s/2 <= 1
        pen_bound = 0.0
        for i in range(n):
            # Left
            val = x[i] - s[i]/2.0
            if val < 0: pen_bound += val*val
            # Right
            val = x[i] + s[i]/2.0 - 1.0
            if val > 0: pen_bound += val*val
            # Bottom
            val = y[i] - s[i]/2.0
            if val < 0: pen_bound += val*val
            # Top
            val = y[i] + s[i]/2.0 - 1.0
            if val > 0: pen_bound += val*val
            
        # 2. Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                req = (s[i] + s[j]) / 2.0
                
                ox = req - dx
                oy = req - dy
                
                # If both > 0, they overlap
                if ox > 0 and oy > 0:
                    pen_overlap += (ox * oy) ** 2
        
        # 3. Objective: Maximize sum(s) -> Minimize -sum(s) + penalties
        sum_s = np.sum(s)
        
        # Weights
        w_bound = 100.0
        w_overlap = 100.0
        
        return -sum_s + w_bound * pen_bound + w_overlap * pen_overlap

    def gradient(v):
        x = v[0:30:3]
        y = v[1:30:3]
        s = v[2:30:3]
        
        grad = np.zeros(30)
        
        # Gradient of -sum(s)
        grad[2:30:3] -= 1.0
        
        # Gradient of boundary penalties
        w_bound = 100.0
        for i in range(n):
            # x
            val_l = x[i] - s[i]/2.0
            if val_l < 0:
                # d/dx (val_l^2) = 2*val_l
                # d/ds (val_l^2) = 2*val_l * (-1/2) = -val_l
                grad[3*i] += w_bound * 2 * val_l
                grad[3*i+2] += w_bound * (-val_l)
            
            val_r = x[i] + s[i]/2.0 - 1.0
            if val_r > 0:
                grad[3*i] += w_bound * 2 * val_r
                grad[3*i+2] += w_bound * val_r
            
            # y
            val_b = y[i] - s[i]/2.0
            if val_b < 0:
                grad[3*i+1] += w_bound * 2 * val_b
                grad[3*i+2] += w_bound * (-val_b)
            
            val_t = y[i] + s[i]/2.0 - 1.0
            if val_t > 0:
                grad[3*i+1] += w_bound * 2 * val_t
                grad[3*i+2] += w_bound * val_t

        # Gradient of overlap penalties
        w_overlap = 100.0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                req = (s[i] + s[j]) / 2.0
                
                ox = req - abs(dx)
                oy = req - abs(dy)
                
                if ox > 0 and oy > 0:
                    # P = (ox * oy)^2
                    # dP/dx_i = 2 * (ox*oy) * oy * d(ox)/dx_i
                    # d(ox)/dx_i = d(req - |x_i - x_j|)/dx_i = -sign(x_i - x_j)
                    
                    sign_dx = 1.0 if dx > 0 else -1.0
                    sign_dy = 1.0 if dy > 0 else -1.0
                    
                    term = 2.0 * (ox * oy)
                    
                    # dP/dx_i
                    d_ox_dx_i = -sign_dx
                    d_oy_dx_i = 0.0
                    grad[3*i] += w_overlap * term * (oy * d_ox_dx_i + ox * d_oy_dx_i)
                    
                    # dP/dx_j
                    d_ox_dx_j = sign_dx
                    grad[3*j] += w_overlap * term * (oy * d_ox_dx_j)
                    
                    # dP/dy_i
                    d_ox_dy_i = 0.0
                    d_oy_dy_i = -sign_dy
                    grad[3*i+1] += w_overlap * term * (oy * d_ox_dy_i + ox * d_oy_dy_i)
                    
                    # dP/dy_j
                    d_oy_dy_j = sign_dy
                    grad[3*j+1] += w_overlap * term * (ox * d_oy_dy_j)
                    
                    # dP/ds_i
                    d_ox_ds_i = 0.5
                    d_oy_ds_i = 0.5
                    grad[3*i+2] += w_overlap * term * (oy * d_ox_ds_i + ox * d_oy_ds_i)
                    
                    # dP/ds_j
                    d_ox_ds_j = 0.5
                    d_oy_ds_j = 0.5
                    grad[3*j+2] += w_overlap * term * (oy * d_ox_ds_j + ox * d_oy_ds_j)

        return grad

    def is_valid(centers, sides, tol=1e-7):
        # Check bounds
        for i in range(len(centers)):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Check overlaps
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                
                t = (s_i + s_j)/2 - tol
                if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                    return False
        return True

    # Initial guesses
    starts = []
    
    # 1. 3x3 grid (9 squares) + 1 small in corner? 
    # Let's just use random valid starts and a grid start.
    
    # Start 1: 3x3 grid, 10th at center of one of them (will push others)
    v1 = np.zeros(30)
    for i in range(9):
        r = i // 3
        c = i % 3
        v1[3*i] = (c + 0.5) / 3.0
        v1[3*i+1] = (r + 0.5) / 3.0
        v1[3*i+2] = 1.0/3.0
    # 10th
    v1[27] = 0.5
    v1[28] = 0.5
    v1[29] = 0.1
    starts.append(v1)
    
    # Start 2: 2x5 grid
    for i in range(10):
        r = i // 5
        c = i % 5
        v1 = np.zeros(30) # Reuse variable name carefully or create new
        # Actually let's make a list of vectors
        pass
    
    v2 = np.zeros(30)
    for i in range(10):
        r = i // 5
        c = i % 5
        v2[3*i] = (c + 0.5) / 5.0
        v2[3*i+1] = (r + 0.5) / 2.0
        v2[3*i+2] = 0.2 # Side 0.2 fits in 0.5 width? 5*0.2=1. Yes. Height 0.5/2=0.25. 0.2 fits.
    starts.append(v2)
    
    # Start 3: Random
    np.random.seed(42)
    for k in range(5):
        v_rand = np.random.rand(30)
        # Scale sides to be small initially
        v_rand[2:30:3] *= 0.2
        v_rand[0:30:3] *= 0.8 + 0.2 * v_rand[2:30:3] # Roughly inside
        v_rand[1:30:3] *= 0.8 + 0.2 * v_rand[2:30:3]
        starts.append(v_rand)

    best_val = -np.inf
    best_v = None
    
    start_time = time.time()
    timeout = 500.0 # Leave 100s margin
    
    for v0 in starts:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = minimize(objective, v0, method='L-BFGS-B', jac=gradient, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if res.success or res.fun < best_val:
                # Extract centers and sides
                v_opt = res.x
                centers = np.column_stack([v_opt[0:30:3], v_opt[1:30:3]])
                sides = v_opt[2:30:3]
                
                # Check validity
                if is_valid(centers, sides):
                    val = np.sum(sides)
                    if val > best_val:
                        best_val = val
                        best_v = v_opt.copy()
        except Exception:
            continue

    if best_v is None:
        # Fallback to a known valid simple config
        # 10 squares of side 0.2 in 2x5 grid
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            c = i % 5
            centers[i][0] = (c + 0.5) / 5.0
            centers[i][1] = (r + 0.5) / 2.0
            sides[i] = 0.2
        return centers, sides

    centers = np.column_stack([best_v[0:30:3], best_v[1:30:3]])
    sides = best_v[2:30:3]
    
    return centers, sides
