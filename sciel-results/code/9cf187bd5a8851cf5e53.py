"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_x = None
    
    # Helper to check validity and calculate score
    def evaluate(x_flat):
        x = x_flat.reshape(N, 3)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        
        # Check boundaries
        if np.any(rs <= 0): return -np.inf
        if np.any(xs - rs < -1e-9) or np.any(1 - xs - rs < -1e-9): return -np.inf
        if np.any(ys - rs < -1e-9) or np.any(1 - ys - rs < -1e-9): return -np.inf
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum**2 - 1e-9:
                    return -np.inf
        return np.sum(rs)

    # Constraints for SLSQP
    # Variables: x0, y0, r0, x1, y1, r1, ...
    def constraints(x_flat):
        x = x_flat.reshape(N, 3)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        cons = []
        
        # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})       # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]}) # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})   # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})# 1 - y_i - r_i >= 0
        
        # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                idx_i = 3*i
                idx_j = 3*j
                def overlap_fun(x, i=i, j=j):
                    dx = x[idx_i] - x[idx_j]
                    dy = x[idx_i+1] - x[idx_j+1]
                    dr = x[idx_i+2] + x[idx_j+2]
                    return dx*dx + dy*dy - dr*dr
                cons.append({'type': 'ineq', 'fun': overlap_fun})
                
        return cons

    # Objective: Maximize sum of radii -> Minimize -sum(r)
    def objective(x_flat):
        r = x_flat[2::3]
        return -np.sum(r)

    def bounds():
        # x, y in [0, 1], r in [0, 0.5] (max possible radius in unit square)
        b = []
        for _ in range(N):
            b.append((0.0, 1.0)) # x
            b.append((0.0, 1.0)) # y
            b.append((1e-6, 0.5)) # r
        return b

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        
        # 1. Grid based with jitter
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(guesses) >= N: break
                x0 = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                y0 = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                r0 = 0.5 / side * 0.8
                guesses.append([x0, y0, r0])
        
        # 2. Random uniform
        x_rand = np.random.uniform(0.1, 0.9, N)
        y_rand = np.random.uniform(0.1, 0.9, N)
        r_rand = np.full(N, 0.05)
        guesses.append(list(zip(x_rand, y_rand, r_rand)))
        
        # 3. Clustered random
        centers = np.random.uniform(0.2, 0.8, (4, 2))
        x_cl, y_cl, r_cl = [], [], []
        for k in range(N):
            c = centers[k % 4]
            x_cl.append(c[0] + np.random.normal(0, 0.05))
            y_cl.append(c[1] + np.random.normal(0, 0.05))
            r_cl.append(0.05)
        guesses.append(list(zip(x_cl, y_cl, r_cl)))
        
        return guesses

    initial_guesses = get_initial_guesses()
    
    # Run SLSQP on multiple starts
    for start_idx, guess in enumerate(initial_guesses):
        if time.time() > deadline - 10: break
        
        x0 = np.array(guess).flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds(), 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or res.fun < -best_score:
                score = evaluate(res.x)
                if score > best_score:
                    best_score = score
                    best_x = res.x.copy()
                    
        except Exception:
            continue

    # If no valid solution found, fallback to simple grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_x = np.array(cells[:N]).flatten()
        best_score = evaluate(best_x)

    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
