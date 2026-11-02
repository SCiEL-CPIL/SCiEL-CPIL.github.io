"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s (or 150s) limit
    
    # Initial guess: grid arrangement
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
    y0 = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
    r0 = np.full(N, 0.5 / side - 1e-3)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
    
    def unflatten(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r
    
    # Objective: maximize sum of radii => minimize -sum(r)
    def objective(v):
        x, y, r = unflatten(v)
        return -np.sum(r)
    
    # Constraints: 
    # 1. Circles inside unit square: x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    # 2. Non-overlapping: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2 for i != j
    # 3. Radii positive: r >= 0
    
    def constraints(v):
        x, y, r = unflatten(v)
        cons = []
        
        # Boundary constraints (lower bounds)
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, y=y, r=r: x - r})
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, y=y, r=r: y - r})
        
        # Boundary constraints (upper bounds)
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, y=y, r=r: 1 - x - r})
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, y=y, r=r: 1 - y - r})
        
        # Pairwise non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2
                })
        
        # Radius positivity
        cons.append({'type': 'ineq', 'fun': lambda v: v[2*N:]})
        
        return cons
    
    # Use SLSQP optimizer
    v0 = flatten(x0, y0, r0)
    
    best_v = v0.copy()
    best_score = objective(v0)
    
    # Try multiple restarts with slight perturbations
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            v_init = v0.copy()
        else:
            # Perturb initial guess
            np.random.seed(42 + restart)
            v_init = v0.copy()
            v_init[0:2*N] += np.random.uniform(-0.05, 0.05, 2*N)
            v_init[2*N:] = np.maximum(v_init[2*N:], 0.01)
        
        try:
            result = minimize(
                objective,
                v_init,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1)] * (2*N) + [(0, 0.5)] * N,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                # Check if solution is valid
                x, y, r = unflatten(result.x)
                valid = True
                for i in range(N):
                    if x[i] - r[i] < -1e-6 or y[i] - r[i] < -1e-6 or x[i] + r[i] > 1 + 1e-6 or y[i] + r[i] > 1 + 1e-6:
                        valid = False
                        break
                    if r[i] < 0:
                        valid = False
                        break
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if dist_sq < (r[i]+r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_v = result.x.copy()
                    best_score = result.fun
        except Exception as e:
            pass
    
    x, y, r = unflatten(best_v)
    # Clip to ensure validity
    r = np.maximum(r, 1e-4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
