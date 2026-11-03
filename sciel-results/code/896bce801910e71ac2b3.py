import numpy as np
from scipy.optimize import minimize
import time

N = 21
START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s buffer

def construct():
    def time_remaining():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    # Helper to generate a hexagonal grid
    def hex_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        x_coords = []
        y_coords = []
        for i in range(n):
            row = i // side
            col = i % side
            x = col + 0.5 * (row % 2)
            y = row * 0.866
            x_coords.append(x)
            y_coords.append(y)
        
        x_max = max(x_coords)
        y_max = max(y_coords)
        x_norm = np.array(x_coords) / (x_max + 1)
        y_norm = np.array(y_coords) / (y_max + 1)
        return x_norm, y_norm

    # Helper to generate a square grid
    def square_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:n])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:n])
        return x, y

    def evaluate_config(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return -np.inf
        if abs(2*(W+H) - 4) > 1e-6:
            return -np.inf
        for i in range(N):
            if r[i] <= 0:
                return -np.inf
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return -np.inf
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return -np.inf
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-6:
                    return -np.inf
        return np.sum(r)

    def greedy_expand(x, y, r, W, H, max_iters=100):
        step = 0.005
        for iteration in range(max_iters):
            if time_remaining() < 5:
                break
            improved = False
            for i in range(N):
                new_r = r[i] + step
                if new_r <= 0:
                    continue
                if x[i] - new_r < 0 or x[i] + new_r > W:
                    continue
                if y[i] - new_r < 0 or y[i] + new_r > H:
                    continue
                valid = True
                for j in range(N):
                    if i == j: 
                        continue
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r[i] = new_r
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-5:
                    break
        return r

    def local_search(x, y, r, W, H, iterations=30):
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        for _ in range(iterations):
            if time_remaining() < 5:
                break
            x_new = x.copy()
            y_new = y.copy()
            # Perturb positions
            x_new += np.random.normal(0, 0.02, N)
            y_new += np.random.normal(0, 0.02, N)
            # Keep radii and try to expand
            r_new = r.copy()
            r_new = greedy_expand(x_new, y_new, r_new, W, H, max_iters=30)
            score = evaluate_config(x_new, y_new, r_new, W, H)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
        return best_x, best_y, best_r, best_score

    # Generate initial candidates
    candidates = []
    
    # 1. Square grid
    x_sq, y_sq = square_grid(N)
    r_sq = np.full(N, 0.03)
    W_sq, H_sq = 1.0, 1.0
    r_sq = greedy_expand(x_sq, y_sq, r_sq, W_sq, H_sq)
    score_sq = evaluate_config(x_sq, y_sq, r_sq, W_sq, H_sq)
    candidates.append((x_sq.copy(), y_sq.copy(), r_sq.copy(), W_sq, H_sq, score_sq))
    
    # 2. Hex grid
    x_hex, y_hex = hex_grid(N)
    r_hex = np.full(N, 0.03)
    W_hex, H_hex = 1.0, 1.0
    r_hex = greedy_expand(x_hex, y_hex, r_hex, W_hex, H_hex)
    score_hex = evaluate_config(x_hex, y_hex, r_hex, W_hex, H_hex)
    candidates.append((x_hex.copy(), y_hex.copy(), r_hex.copy(), W_hex, H_hex, score_hex))
    
    # 3. Random jittered square
    np.random.seed(42)
    x_rand = x_sq + np.random.normal(0, 0.03, N)
    y_rand = y_sq + np.random.normal(0, 0.03, N)
    r_rand = np.full(N, 0.03)
    W_rand, H_rand = 1.0, 1.0
    r_rand = greedy_expand(x_rand, y_rand, r_rand, W_rand, H_rand)
    score_rand = evaluate_config(x_rand, y_rand, r_rand, W_rand, H_rand)
    candidates.append((x_rand.copy(), y_rand.copy(), r_rand.copy(), W_rand, H_rand, score_rand))
    
    # Sort by score
    candidates.sort(key=lambda c: c[5], reverse=True)
    
    best_x, best_y, best_r, best_W, best_H, best_score = candidates[0]
    
    # Run local search on best candidate
    if time_remaining() > 10:
        best_x, best_y, best_r, best_score = local_search(
            best_x, best_y, best_r, best_W, best_H, iterations=20
        )
    
    # Run SLSQP on the best candidate with time limit
    def obj(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        return -np.sum(r)
        
    def jac(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
        
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

    x_start = np.concatenate([best_x, best_y, best_r, [best_W, best_H]])
    
    if time_remaining() > 15:
        try:
            res = minimize(obj, x_start, jac=jac, method='SLSQP', 
                          constraints=constraints, 
                          options={'maxiter': 150, 'ftol': 1e-7})
            if res.success:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                
                score_final = evaluate_config(x_final, y_final, r_final, W_final, H_final)
                if score_final > best_score:
                    best_x, best_y, best_r, best_W, best_H = x_final, y_final, r_final, W_final, H_final
                    best_score = score_final
        except Exception:
            pass
    
    # Final check and return
    if best_score == -np.inf:
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 0.02)
        W, H = 1.0, 1.0
        return np.column_stack([x, y, r])
        
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
