import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_solution = None
    
    # Helper to check validity and calculate score
    def evaluate(sol):
        if sol is None:
            return -np.inf
        x = sol[0::3]
        y = sol[1::3]
        r = sol[2::3]
        
        # Check bounds
        if np.any(r <= 0) or np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
            return -np.inf
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return -np.inf
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum_sq = (r[i] + r[j])**2
                if dist_sq < rad_sum_sq - 1e-9:
                    return -np.inf
        return np.sum(r)

    # Objective function: minimize negative sum of radii
    def objective(v):
        r = v[2::3]
        return -np.sum(r)

    # Constraints for SLSQP
    def make_constraints():
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]}) # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i >= 0
            
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def dist_fun(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': dist_fun})
        
        return cons

    # Bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    
    constraints = make_constraints()
    
    # Define initial guesses
    # 1. Jittered grid
    def get_initial_grid(jitter):
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x0[count] = (i + 0.5) / side + (np.random.rand() - 0.5) * 0.05 * jitter
                y0[count] = (j + 0.5) / side + (np.random.rand() - 0.5) * 0.05 * jitter
                r0[count] = 0.5 / side - 0.01
                count += 1
        # Clip to bounds
        x0 = np.clip(x0, 0.1, 0.9)
        y0 = np.clip(y0, 0.1, 0.9)
        r0 = np.clip(r0, 0.01, 0.4)
        v0 = np.zeros(3*N)
        v0[0::3] = x0
        v0[1::3] = y0
        v0[2::3] = r0
        return v0

    # Run multiple restarts
    restarts = 0
    max_restarts = 20
    
    while time.time() - start_time < timeout and restarts < max_restarts:
        restarts += 1
        # Vary initial guess
        v0 = get_initial_grid(jitter=1.0)
        
        # Optimization
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                sol = res.x
                score = evaluate(sol)
                if score > best_score:
                    best_score = score
                    best_solution = sol
            else:
                # Even if not successful, check if the current point is feasible and better
                # SLSQP often returns feasible points even if it hits maxiter
                sol = res.x
                score = evaluate(sol)
                if score > best_score:
                    best_score = score
                    best_solution = sol
        except Exception:
            continue

    # If no valid solution found, fall back to a very small grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.05
                cells.append([x, y, r])
                count += 1
        best_solution = np.array(cells)
    
    # Final safety check and formatting
    # Ensure the returned array is exactly shape (N, 3)
    if len(best_solution) != 3*N:
        # Fallback if something went wrong with shape
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.05
                cells.append([x, y, r])
                count += 1
        best_solution = np.array(cells)
        
    # Reshape to (N, 3)
    result = best_solution.reshape(N, 3)
    
    # Final tiny shrink to ensure floating point safety
    # Check if any constraint is violated by a tiny amount
    x, y, r = result[:,0], result[:,1], result[:,2]
    
    # Adjust radii down slightly if necessary to satisfy boundary
    r = np.minimum(r, x)
    r = np.minimum(r, 1-x)
    r = np.minimum(r, y)
    r = np.minimum(r, 1-y)
    r = np.maximum(r, 1e-6)
    
    # Check overlaps and shrink if needed (simple pairwise check)
    # This is a heuristic fix. A more rigorous approach would be to re-optimize with tighter constraints,
    # but for a final safety net, we just shrink radii globally if a violation is detected.
    # However, shrinking globally might reduce score. 
    # Better: Just return the optimized result. SLSQP with ftol=1e-9 should be safe.
    # We'll do a final strict check. If invalid, we shrink radii by 0.1%.
    
    valid = True
    for i in range(N):
        if r[i] > x[i] or r[i] > 1-x[i] or r[i] > y[i] or r[i] > 1-y[i]:
            valid = False
            break
        for j in range(i+1, N):
            d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            rs = r[i] + r[j]
            if d2 < rs*rs - 1e-10:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Shrink all radii by 5% to ensure feasibility
        r = r * 0.95
        
    result[:, 2] = r
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# The prompt asks to keep the function names and signatures.
# I will place the code inside the block as requested by the structure, 
# but since the whole file is the program, I will just provide the full code.
# Wait, the prompt says "Rewrite the program from scratch... Output the COMPLETE program".
# And "Keep the function names...".
# The seed had N=26 and construct().
# I will output the full script.
# EVOLVE-BLOCK-END
