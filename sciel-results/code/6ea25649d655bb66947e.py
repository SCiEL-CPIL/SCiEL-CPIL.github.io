"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # We will use a log-barrier interior point method to maximize sum of radii
    # subject to:
    # 1. x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    # 2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    
    # The optimization variables are:
    #   z = [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    # Total dimension: 3*N = 78
    
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    # Constraints (inequalities, must be >= 0 for scipy's 'ineq'):
    #   g_k(z) >= 0
    
    # Boundary constraints:
    #   x_i - r_i >= 0
    #   1 - x_i - r_i >= 0
    #   y_i - r_i >= 0
    #   1 - y_i - r_i >= 0
    #   r_i >= 0 (we can add small epsilon)
    
    # Non-overlap constraints:
    #   (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # This is a non-convex problem. We'll use multiple random restarts.
    
    def objective(z):
        r = z[2::3]
        return -np.sum(r)
    
    def constraints(z):
        cons = []
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary constraints
        cons.append({'type': 'ineq', 'fun': lambda z: z[0::3] - z[2::3]})  # x_i - r_i
        cons.append({'type': 'ineq', 'fun': lambda z: 1.0 - z[0::3] - z[2::3]})  # 1 - x_i - r_i
        cons.append({'type': 'ineq', 'fun': lambda z: z[1::3] - z[2::3]})  # y_i - r_i
        cons.append({'type': 'ineq', 'fun': lambda z: 1.0 - z[1::3] - z[2::3]})  # 1 - y_i - r_i
        cons.append({'type': 'ineq', 'fun': lambda z: z[2::3] - 1e-6})  # r_i >= eps
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                def nonoverlap(z, i=i, j=j):
                    xi = z[3*i]
                    yi = z[3*i+1]
                    ri = z[3*i+2]
                    xj = z[3*j]
                    yj = z[3*j+1]
                    rj = z[3*j+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': nonoverlap})
        
        return cons
    
    def bounds():
        b = []
        for i in range(N):
            b.append((1e-4, 1.0 - 1e-4))  # x_i
            b.append((1e-4, 1.0 - 1e-4))  # y_i
            b.append((1e-4, 0.5))  # r_i
        return b
    
    best_value = 0.0
    best_z = None
    
    # Initial feasible solution: grid of small circles
    def make_initial():
        z = np.zeros(3*N)
        # Place circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.9
                z[3*count] = cx
                z[3*count+1] = cy
                z[3*count+2] = r
                count += 1
        return z
    
    z0 = make_initial()
    
    # Try a few random restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            z_init = z0.copy()
        else:
            # Perturb the initial solution
            z_init = z0.copy()
            for i in range(N):
                z_init[3*i] += np.random.uniform(-0.05, 0.05)
                z_init[3*i+1] += np.random.uniform(-0.05, 0.05)
                z_init[3*i+2] = max(1e-4, z_init[3*i+2] * np.random.uniform(0.8, 1.2))
                # Clip to feasible range
                z_init[3*i] = np.clip(z_init[3*i], z_init[3*i+2] + 1e-4, 1 - z_init[3*i+2] - 1e-4)
                z_init[3*i+1] = np.clip(z_init[3*i+1], z_init[3*i+2] + 1e-4, 1 - z_init[3*i+2] - 1e-4)
        
        # Check if initial is feasible
        def is_feasible(z, tol=1e-6):
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            # Boundary
            if np.any(x < r - tol) or np.any(x > 1 - r + tol):
                return False
            if np.any(y < r - tol) or np.any(y > 1 - r + tol):
                return False
            if np.any(r < 1e-6):
                return False
            # Non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    if dist_sq < (r[i]+r[j])**2 - tol:
                        return False
            return True
        
        if not is_feasible(z_init, tol=1e-5):
            # Try to project to feasibility by reducing radii
            for i in range(N):
                r_max = min(z_init[3*i] - 1e-4, 1 - z_init[3*i] - 1e-4,
                            z_init[3*i+1] - 1e-4, 1 - z_init[3*i+1] - 1e-4)
                z_init[3*i+2] = max(1e-4, min(z_init[3*i+2], r_max))
        
        try:
            res = minimize(objective, z_init, method='SLSQP', bounds=bounds(),
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or (res.fun < 0 and is_feasible(res.x, tol=1e-5)):
                value = -res.fun
                if value > best_value:
                    best_value = value
                    best_z = res.x.copy()
        except Exception as e:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # If we found a good solution, use it; otherwise fall back to initial
    if best_z is not None:
        z = best_z
    else:
        z = z0
    
    # Final polish: try to improve with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, z, method='SLSQP', bounds=bounds(),
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success or (res2.fun < 0 and is_feasible(res2.x, tol=1e-6)):
                value2 = -res2.fun
                if value2 > best_value:
                    best_value = value2
                    z = res2.x
        except:
            pass
    
    # Ensure final solution is feasible
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Clip to boundaries
    for i in range(N):
        r[i] = max(1e-6, min(r[i], 0.5))
        x[i] = np.clip(x[i], r[i] + 1e-6, 1 - r[i] - 1e-6)
        y[i] = np.clip(y[i], r[i] + 1e-6, 1 - r[i] - 1e-6)
    
    # Check for overlaps and resolve by reducing radii if needed
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            required = r[i] + r[j]
            if dist < required:
                # Reduce the smaller radius
                scale = dist / required * 0.999
                if r[i] < r[j]:
                    r[i] *= scale
                else:
                    r[j] *= scale
    
    # Final feasibility check
    for i in range(N):
        r[i] = max(1e-6, min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) - 1e-6))
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j]:
                # Further reduce
                scale = dist / (r[i] + r[j]) * 0.999
                if r[i] < r[j]:
                    r[i] *= scale
                else:
                    r[j] *= scale
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
