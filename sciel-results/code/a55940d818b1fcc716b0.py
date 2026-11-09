import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Objective function with penalty for violations
    # vars: [w, x_0...x_N, y_0...y_N, r_0...r_N]
    def objective(vars):
        w = vars[0]
        h = 2.0 - w
        
        # Bounds check for w
        if w <= 0 or h <= 0:
            return 1e10
        if w < 0.1 or w > 1.9:
            return 1e10

        x = vars[1:1+N]
        y = vars[1+N:1+2*N]
        r = vars[1+2*N:1+3*N]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalty for boundary violations
        # x_i must be in [r_i, w-r_i]
        # y_i must be in [r_i, h-r_i]
        pen_boundary = 0.0
        for i in range(N):
            if x[i] < r[i]:
                pen_boundary += (r[i] - x[i])**2
            elif x[i] > w - r[i]:
                pen_boundary += (x[i] - (w - r[i]))**2
            
            if y[i] < r[i]:
                pen_boundary += (r[i] - y[i])**2
            elif y[i] > h - r[i]:
                pen_boundary += (y[i] - (h - r[i]))**2
                
            # Radius must be positive
            if r[i] < 0:
                pen_boundary += r[i]**2

        # Penalty for overlap violations
        # dist^2 >= (r_i + r_j)^2
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx*dx + dy*dy
                sum_r_ij = r[i] + r[j]
                if dist2 < sum_r_ij**2:
                    # Penalty is proportional to the amount of overlap squared
                    # Using sqrt to make it smooth: (sum_r - dist)^2
                    # But sqrt is expensive. 
                    # Approximation: (sum_r^2 - dist2)^2 is smooth but scales badly.
                    # Let's use (sum_r - sqrt(dist2))**2 if we can, but sqrt is slow.
                    # Alternative: Just use the difference in squared distances?
                    # No, that's not a good metric.
                    # Let's use: max(0, sum_r - sqrt(dist2))**2
                    # Since we are in a penalty method, we can afford some computation.
                    # However, N=21, pairs=210. 210 sqrts is fine.
                    dist = np.sqrt(dist2)
                    if sum_r_ij > dist:
                        pen_overlap += (sum_r_ij - dist)**2

        # Total penalty
        # We want to minimize -sum_r + penalty
        # If penalty is 0, we maximize sum_r.
        # We need a large enough weight for penalty to ensure feasibility.
        # Let's use a weight of 100.
        weight = 100.0
        return -sum_r + weight * (pen_boundary + pen_overlap)

    def random_init():
        # Random initialization
        # Try a mix of square-ish and rectangular boxes
        w = np.random.uniform(0.8, 1.2)
        h = 2.0 - w
        
        # Initialize circles in a random grid or random positions
        # Let's try random positions within the box
        x = np.random.uniform(0.1, w-0.1, N)
        y = np.random.uniform(0.1, h-0.1, N)
        # Initial radii: small enough to likely fit
        # Max possible radius is min(w,h)/2, but we need to pack N of them.
        # Approx area packing: N * pi * r^2 <= w * h
        # r <= sqrt(w*h / (N * pi))
        r_max_est = np.sqrt(w * h / (N * np.pi))
        # Add some randomness
        r = np.random.uniform(0.5 * r_max_est, r_max_est, N)
        
        return np.concatenate(([w], x, y, r))

    best_val = -1e10
    best_result = None
    
    # Number of restarts
    n_restarts = 30
    
    # Options for Nelder-Mead
    options = {'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
    
    for _ in range(n_restarts):
        vars0 = random_init()
        try:
            res = minimize(objective, vars0, method='Nelder-Mead', options=options)
            
            # Check if the result is valid
            w = res.x[0]
            h = 2.0 - w
            x = res.x[1:1+N]
            y = res.x[1+N:1+2*N]
            r = res.x[1+2*N:1+3*N]
            
            valid = True
            # Check boundaries
            for i in range(N):
                if r[i] <= 0:
                    valid = False
                    break
                if x[i] < r[i] - 1e-7 or x[i] > w - r[i] + 1e-7:
                    valid = False
                    break
                if y[i] < r[i] - 1e-7 or y[i] > h - r[i] + 1e-7:
                    valid = False
                    break
            
            # Check overlaps
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        sum_r = r[i] + r[j]
                        if dist2 < sum_r**2 - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = res.x
                    
        except Exception:
            continue

    if best_result is not None:
        w = best_result[0]
        x = best_result[1:1+N]
        y = best_result[1+N:1+2*N]
        r = best_result[1+2*N:1+3*N]
        
        # Final cleanup/verification
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > (2-w) - r[i] or r[i] <= 0:
                valid = False
                break
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                if dist2 < sum_r**2 - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            return np.column_stack((x, y, r))
        else:
            # Fallback
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
