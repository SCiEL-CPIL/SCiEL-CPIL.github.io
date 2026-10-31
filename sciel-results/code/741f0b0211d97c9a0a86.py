"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    # Variables: x[0:N], y[0:N], r[0:N]
    # Constraints:
    #   0 <= x_i <= 1 - r_i  =>  x_i + r_i <= 1, 1 - x_i - r_i >= 0
    #   0 <= y_i <= 1 - r_i  =>  y_i + r_i <= 1, 1 - y_i - r_i >= 0
    #   r_i >= 0
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i != j
    
    def objective(v):
        r = v[2*N:]
        return -np.sum(r)
    
    def constraints_def():
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - 1e-10})  # x_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i + N] - 1e-10})  # y_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*N+i] - 1e-10})  # x_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i + N] - v[2*N+i] - 1e-10})  # y_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i] - 1e-10})  # r_i >= 0
        
        # Pairwise non-overlap
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[i+N]-v[j+N])**2 - (v[2*N+i]+v[2*N+j])**2
                })
        return cons
    
    constraints = constraints_def()
    
    best_result = None
    best_value = -np.inf
    
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(3*N)
    for i in range(N):
        col = i % side
        row = i // side
        x0[i] = (col + 0.5) / side
        x0[i + N] = (row + 0.5) / side
        x0[2*N + i] = 0.5 / side - 0.01
    
    # Try multiple random restarts
    n_restarts = 20
    attempts = 0
    
    for restart in range(n_restarts):
        if time_left() < 5:
            break
        
        if restart == 0:
            v0 = x0.copy()
        else:
            # Perturb the initial guess
            v0 = x0.copy()
            noise_scale = 0.05
            v0[:N] += np.random.randn(N) * noise_scale
            v0[N:2*N] += np.random.randn(N) * noise_scale
            # Keep radii positive and reasonable
            v0[2*N:] = np.maximum(v0[2*N:] + np.random.randn(N) * 0.02, 0.01)
            # Clip to feasible region approximately
            for i in range(N):
                v0[i] = np.clip(v0[i], 0.01, 0.99)
                v0[i + N] = np.clip(v0[i + N], 0.01, 0.99)
                v0[2*N + i] = np.clip(v0[2*N + i], 0.01, min(1 - v0[i], 1 - v0[i + N]))
        
        try:
            res = minimize(
                objective,
                v0,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1)] * N + [(0, 1)] * N + [(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            attempts += 1
        except Exception:
            continue
        
        if res.success or res.fun < best_value:
            v = res.x
            # Verify feasibility
            feasible = True
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            # Check boundaries
            if np.any(x < -1e-6) or np.any(y < -1e-6) or np.any(r < -1e-6):
                feasible = False
            if np.any(x + r > 1 + 1e-6) or np.any(y + r > 1 + 1e-6):
                feasible = False
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    if dist_sq < (r[i] + r[j])**2 - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = v.copy()
        
        if time_left() < 10:
            break
    
    # If we found a result, return it; otherwise return the initial grid
    if best_result is not None:
        x = best_result[:N]
        y = best_result[N:2*N]
        r = best_result[2*N:]
    else:
        # Fallback to grid
        x = np.array([(i % side + 0.5) / side for i in range(N)])
        y = np.array([(i // side + 0.5) / side for i in range(N)])
        r = np.full(N, 0.5 / side - 0.01)
    
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
