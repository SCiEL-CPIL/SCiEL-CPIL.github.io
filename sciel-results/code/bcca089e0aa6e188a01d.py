import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for overhead and return

    best_score = -np.inf
    best_result = None

    # Variables: x[0..N-1], y[0..N-1], r[0..N-1], W, H
    # Total variables: 3*N + 2

    def constraints_def(x):
        # x is flattened array [x0..xN-1, y0..yN-1, r0..rN-1, W, H]
        cons = []
        
        # Perimeter constraint: 2(W+H) <= 4  =>  W+H <= 2
        # Inequality constraint for SLSQP: func(x) >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: 2 - (x[3*N] + x[3*N+1])})
        
        # Boundary constraints for each circle i
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
            # W - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N] - x[i] - x[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
            # H - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N+1] - x[N+i] - x[2*N+i]})
            # r_i >= 0 (small positive to avoid degenerate)
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i] - 1e-6})
            
        # Non-overlap constraints for each pair i, j
        for i in range(N):
            for j in range(i+1, N):
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: 
                        (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
                })
                
        return cons

    def objective(x):
        # Maximize sum of radii -> Minimize -sum(r)
        rs = x[2*N:3*N]
        return -np.sum(rs)

    def is_valid(x, tol=1e-5):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        if W <= 0 or H <= 0:
            return False
        if 2*(W+H) > 4 + tol:
            return False
        
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > W - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > H - rs[i] + tol:
                return False
            if rs[i] < 1e-6:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - tol:
                    return False
        return True

    def get_score(x):
        if not is_valid(x):
            return -np.inf
        rs = x[2*N:3*N]
        return np.sum(rs)

    # Initializations
    def hex_init(W, H):
        # Generate hexagonal packing in W x H
        # Approximate radius for hex packing in square area A: r ~ sqrt(A/N) * constant
        # Let's just place them in a grid and perturb
        # We want to fill W x H
        # Use a simple grid for initialization, then let optimizer work
        side = int(np.ceil(np.sqrt(N)))
        # Scale to fit W, H
        # Center of cells
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Simple grid placement
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        dx = W / cols
        dy = H / rows
        
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N: break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                # Initial radius estimate
                # Max possible radius in this cell is min(dx, dy)/2
                # But we need to account for neighbors.
                # Start with a small radius, optimizer will grow them
                rs[k] = 0.5 * min(dx, dy) * 0.8
                k += 1
                
        return np.concatenate([xs, ys, rs, [W, H]])

    def random_init(W, H):
        xs = np.random.uniform(0.1, W-0.1, N)
        ys = np.random.uniform(0.1, H-0.1, N)
        rs = np.random.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs, [W, H]])

    def relax_radii(x, n_iter=10, tol=1e-8):
        """
        Iteratively try to increase each circle's radius given fixed positions.
        This acts as a refinement step after SLSQP.
        """
        x = x.copy()
        xs = x[:N].copy()
        ys = x[N:2*N].copy()
        rs = x[2*N:3*N].copy()
        W = x[3*N]
        H = x[3*N+1]
        
        for it in range(n_iter):
            improved = False
            for i in range(N):
                # Max radius from boundaries
                r_bound = min(xs[i], W - xs[i], ys[i], H - ys[i])
                
                # Max radius from other circles
                r_max = r_bound
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    # Circle i can be at most d - r[j] in radius (if touching j)
                    r_candidate = d - rs[j]
                    if r_candidate < r_max:
                        r_max = r_candidate
                
                r_max = max(r_max, 1e-6)
                if r_max > rs[i] + tol:
                    rs[i] = r_max
                    improved = True
            
            if not improved:
                break
        
        x[2*N:3*N] = rs
        return x

    def local_search(x, step_sizes=[0.05, 0.02, 0.01], n_steps=10):
        """
        Simple local search: try moving circles slightly and see if it allows
        larger radii (via relax_radii).
        """
        best_x = x.copy()
        best_score = get_score(best_x)
        
        for step in step_sizes:
            for _ in range(n_steps):
                improved = False
                for i in range(N):
                    # Try small moves in each direction
                    base_score = get_score(best_x)
                    if base_score == -np.inf:
                        base_score = 0 # Avoid issues if current is invalid, though we assume valid input
                    
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            new_x = best_x.copy()
                            new_x[i] += dx
                            new_x[N+i] += dy
                            
                            # Keep within bounds roughly
                            W = new_x[3*N]
                            H = new_x[3*N+1]
                            r_i = new_x[2*N+i]
                            if new_x[i] < r_i or new_x[i] > W - r_i:
                                continue
                            if new_x[N+i] < r_i or new_x[N+i] > H - r_i:
                                continue
                                
                            # Relax radii
                            new_x_relaxed = relax_radii(new_x, n_iter=5)
                            score = get_score(new_x_relaxed)
                            
                            if score > base_score + 1e-8:
                                best_x = new_x_relaxed
                                base_score = score
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    break
            # Update best_score for next step size
            best_score = get_score(best_x)
            
        return best_x

    attempts = 0
    max_attempts = 80 # Cap to ensure we don't loop forever if one is very slow
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Vary aspect ratio
        # W + H = 2
        # W in [0.5, 1.5] roughly (avoid too extreme)
        # Bias towards square initially, then explore
        if attempts < 10:
            W = 1.0
        else:
            W = np.random.uniform(0.7, 1.3)
        H = 2.0 - W
        
        # Choose initialization strategy
        if attempts % 4 == 0:
            x0 = random_init(W, H)
        else:
            x0 = hex_init(W, H)
            # Add small noise to escape symmetry
            x0[:3*N] += np.random.normal(0, 0.05, 3*N)
            
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           bounds=[(1e-4, None)] * (3*N) + [(0.5, 1.5), (0.5, 1.5)],
                           constraints=constraints_def,
                           options={'maxiter': 150, 'ftol': 1e-8})
            
            x_opt = res.x
            score = get_score(x_opt)
            
            # If SLSQP result is valid, try to refine it
            if score > -np.inf:
                # 1. Relax radii
                x_relaxed = relax_radii(x_opt, n_iter=20)
                score_relaxed = get_score(x_relaxed)
                if score_relaxed > score:
                    x_opt = x_relaxed
                    score = score_relaxed
                
                # 2. Local search if time permits
                if time.time() < deadline - 10:
                    x_ls = local_search(x_opt, step_sizes=[0.02, 0.01], n_steps=5)
                    score_ls = get_score(x_ls)
                    if score_ls > score:
                        x_opt = x_ls
                        score = score_ls
            
            if score > best_score:
                best_score = score
                best_result = x_opt

        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like solution if optimization failed completely
        # Grid in 1x1 square (perimeter 4)
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract final result
    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    
    # Final post-processing: Ensure strict validity by shrinking radii slightly if needed
    # The optimizer might leave them touching exactly.
    # We can try to find a global scale factor or just return as is if valid.
    # The check is_valid uses 1e-5 tolerance.
    
    # If the result is valid, return it.
    # The evaluator expects array of shape (n, 3): [x, y, r]
    result = np.column_stack((xs, ys, rs))
    
    return result
