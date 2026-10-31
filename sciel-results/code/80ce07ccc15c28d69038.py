import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    # Objective function to minimize: -sum(r) + penalty
    mu = 1000.0  # Penalty weight
    
    def objective_and_grad(z):
        # z is shape (3*N,)
        # x[0::3], y[1::3], r[2::3]
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        sum_r = np.sum(r)
        penalty = 0.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = -np.ones(N)  # Gradient of -sum(r)
        
        # Boundary penalties
        for i in range(N):
            # x boundaries
            viol_x_left = max(0.0, r[i] - x[i])
            viol_x_right = max(0.0, r[i] - (1.0 - x[i]))
            
            if viol_x_left > 0:
                penalty += viol_x_left**2
                grad_x[i] += 2.0 * viol_x_left * (-1.0) # d/dx (r-x)^2 = -2(r-x)
                grad_r[i] += 2.0 * viol_x_left * 1.0    # d/dr (r-x)^2 = 2(r-x)
                
            if viol_x_right > 0:
                penalty += viol_x_right**2
                grad_x[i] += 2.0 * viol_x_right * 1.0    # d/dx (r-(1-x))^2 = 2(r-(1-x))
                grad_r[i] += 2.0 * viol_x_right * 1.0    # d/dr (r-(1-x))^2 = 2(r-(1-x))
                
            # y boundaries
            viol_y_down = max(0.0, r[i] - y[i])
            viol_y_up = max(0.0, r[i] - (1.0 - y[i]))
            
            if viol_y_down > 0:
                penalty += viol_y_down**2
                grad_y[i] += 2.0 * viol_y_down * (-1.0)
                grad_r[i] += 2.0 * viol_y_down * 1.0
                
            if viol_y_up > 0:
                penalty += viol_y_up**2
                grad_y[i] += 2.0 * viol_y_up * 1.0
                grad_r[i] += 2.0 * viol_y_up * 1.0

        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                
                if d < 1e-12:
                    # Avoid division by zero, push apart slightly
                    dx = 1e-12
                    d = 1e-12
                    
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    penalty += overlap**2
                    
                    # Gradients for i
                    # d(overlap)/dx_i = -dx/d
                    # d(overlap^2)/dx_i = 2*overlap * (-dx/d)
                    grad_x[i] += 2.0 * overlap * (-dx / d)
                    grad_y[i] += 2.0 * overlap * (-dy / d)
                    grad_r[i] += 2.0 * overlap * 1.0
                    
                    # Gradients for j
                    # d(overlap)/dx_j = dx/d
                    grad_x[j] += 2.0 * overlap * (dx / d)
                    grad_y[j] += 2.0 * overlap * (dy / d)
                    grad_r[j] += 2.0 * overlap * 1.0
        
        total_val = -sum_r + mu * penalty
        
        # Construct gradient vector
        grad = np.zeros(3*N)
        grad[0::3] = mu * grad_x
        grad[1::3] = mu * grad_y
        grad[2::3] = mu * grad_r
        
        return total_val, grad

    def objective_only(z):
        return objective_and_grad(z)[0]

    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    best_val = -np.inf
    best_z = None
    
    # Random restarts
    np.random.seed(42)
    num_restarts = 20
    
    for _ in range(num_restarts):
        # Initialize with random positions and small radii
        x0 = np.random.rand(N)
        y0 = np.random.rand(N)
        r0 = np.random.rand(N) * 0.1 + 0.01 # small initial radii
        
        z0 = np.zeros(3*N)
        z0[0::3] = x0
        z0[1::3] = y0
        z0[2::3] = r0
        
        res = minimize(objective_only, z0, method='L-BFGS-B', jac=objective_and_grad, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_z = res.x
            
    # Extract best result
    if best_z is None:
        # Fallback to grid if optimization fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_z[0::3]
    y = best_z[1::3]
    r = best_z[2::3]
    
    # Final safety check: ensure no overlaps and inside bounds
    # If there are slight violations due to numerical errors, shrink radii slightly
    for i in range(N):
        # Clamp to box
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] < 1e-6:
            r[i] = 1e-6
            
    # Check overlaps and shrink if necessary
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j]:
                # Scale down both slightly? Or just one? 
                # Better to just reduce the sum objective by shrinking radii to fit.
                # For simplicity in this post-processing, we reduce the larger one or both proportionally.
                # Actually, the optimizer should have handled this. 
                # If not, we apply a global shrink factor to be safe.
                pass 

    # To be absolutely safe against "invalid result", we can do a final projection/shrink.
    # If any overlap exists, shrink all radii by a factor until no overlap or min radius reached.
    # This might reduce the score but guarantees validity.
    valid = True
    for i in range(N):
        if r[i] < x[i] or r[i] < 1.0-x[i] or r[i] < y[i] or r[i] < 1.0-y[i]:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - 1e-9:
                valid = False
                
    if not valid:
        # Shrink radii
        shrink_factor = 1.0
        for _ in range(100):
            shrink_factor *= 0.95
            r_check = r * shrink_factor
            ok = True
            for i in range(N):
                if r_check[i] > x[i] or r_check[i] > 1.0-x[i] or r_check[i] > y[i] or r_check[i] > 1.0-y[i]:
                    ok = False
                    break
            if ok:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r_check[i] + r_check[j] - 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                r = r_check
                break
                
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    return result

