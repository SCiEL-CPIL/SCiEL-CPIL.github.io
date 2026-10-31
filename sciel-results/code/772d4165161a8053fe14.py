import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial setup
    # 6x6 grid, take 26
    side = 6
    idxs = [i for i in range(side*side)][:N]
    init_pos = np.array([((i % side + 0.5)/side, (i // side + 0.5)/side) for i in idxs])
    init_r = np.full(N, 0.05)
    
    best_val = 0
    best_res = None
    
    # Multi-start
    n_starts = 5
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        # Jitter
        pos = init_pos + np.random.uniform(-0.02, 0.02, size=(N, 2))
        pos = np.clip(pos, 0.01, 0.99)
        r = np.full(N, 0.05)
        
        # Optimization Loop
        for iter in range(50):
            if time.time() > deadline:
                break
                
            # 1. Update Radii via LP
            r = solve_lp_radii(pos, r)
            
            # 2. Update Positions via SLSQP
            pos_new = optimize_positions_slsqp(pos, r)
            
            if np.allclose(pos, pos_new):
                break
            pos = pos_new
            
        # Final LP to get exact radii for these positions
        r_final = solve_lp_radii(pos, r)
        
        val = np.sum(r_final)
        if val > best_val:
            best_val = val
            best_res = np.column_stack((pos, r_final))
            
    if best_res is None:
        # Fallback to trivial
        return np.array([((i % 6 + 0.5)/6, (i // 6 + 0.5)/6, 0.08) for i in range(N)])
        
    return best_res

def solve_lp_radii(pos, r_init):
    # Max sum(r)
    # s.t. r_i + r_j <= dist(i,j)
    #      r_i <= dist_to_wall(i)
    #      r_i >= 0
    
    c = -np.ones(N) # Minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    d = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
    w = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1]))
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d[i, j])
            
    # Wall constraints: r_i <= w_i
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(w[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * N, method='highs')
    
    if res.success:
        return res.x
    else:
        return r_init # Fallback

def optimize_positions_slsqp(pos, r):
    # Maximize delta
    # s.t. d_ij - (r_i + r_j) >= delta
    #      w_i - r_i >= delta
    #      0 <= pos <= 1
    
    x0 = np.concatenate([pos[:, 0], pos[:, 1]])
    
    def objective(x):
        p = np.column_stack((x[:N], x[N:]))
        d = np.linalg.norm(p[:, np.newaxis] - p[np.newaxis, :], axis=2)
        w = np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]), np.minimum(p[:, 1], 1 - p[:, 1]))
        
        # Min slacks
        slacks_pair = d - np.outer(r, np.ones(N)) - np.outer(np.ones(N), r)
        # We want min over i<j of slacks_pair[i,j]
        # And min over i of w_i - r_i
        
        min_pair_slack = np.inf
        for i in range(N):
            for j in range(i+1, N):
                min_pair_slack = min(min_pair_slack, slacks_pair[i, j])
                
        min_wall_slack = np.min(w - r)
        
        return -min(min_pair_slack, min_wall_slack) # Maximize min slack
        
    constraints = []
    # Bounds are handled by bounds parameter
    # We don't need explicit constraints if we just maximize the min slack directly?
    # Yes, SLSQP can handle objective function directly.
    
    bounds = [(0, 1)] * (2 * N)
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-6})
    
    if res.success:
        p_new = np.column_stack((res.x[:N], res.x[N:]))
        return p_new
    else:
        return pos
