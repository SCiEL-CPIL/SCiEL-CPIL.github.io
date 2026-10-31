import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Grid initialization
    side = 5
    xs = np.array([(i + 0.5) * 2 for i in range(side)])
    ys = np.array([(j + 0.5) * 2 for j in range(side)])
    X, Y = np.meshgrid(xs, ys)
    x0 = X.flatten()[:N]
    y0 = Y.flatten()[:N]
    
    best_r = 0.1 # Seed radius
    best_x = x0
    best_y = y0
    
    def objective_penalty(v):
        x = v[:N]
        y = v[N:2*N]
        W = np.max(x) - np.min(x)
        H = np.max(y) - np.min(y)
        
        pen = 0.0
        # Vectorized overlap check for speed?
        # N=21 is small, loop is fine
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                if d2 < 4.0:
                    d = np.sqrt(d2)
                    pen += (2.0 - d)**2
        return W + H + 100.0 * pen

    for _ in range(10):
        noise = np.random.normal(0, 0.5, 2*N)
        v0 = np.concatenate([x0 + noise[:N], y0 + noise[N:]])
        res = minimize(objective_penalty, v0, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-14})
        
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        
        W_c = np.max(x_opt) - np.min(x_opt)
        H_c = np.max(y_opt) - np.min(y_opt)
        S = W_c + H_c
        
        # Check for overlap
        min_d = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = x_opt[i] - x_opt[j]
                dy = y_opt[i] - y_opt[j]
                min_d = min(min_d, np.sqrt(dx*dx + dy*dy))
        
        # Theoretical max r from box
        r_box = 2.0 / (S + 4.0)
        # Max r from overlap
        r_overlap = min_d / 2.0
        
        r_candidate = min(r_box, r_overlap)
        
        # Add small epsilon to ensure strict disjoint interiors
        r_candidate *= 0.9999
        
        if r_candidate > best_r:
            best_r = r_candidate
            best_x = x_opt
            best_y = y_opt

    # Scale final solution
    x_min = np.min(best_x)
    y_min = np.min(best_y)
    
    x_final = (best_x - x_min) * best_r + best_r
    y_final = (best_y - y_min) * best_r + best_r
    
    circles = np.column_stack((x_final, y_final, np.full(N, best_r)))
    
    return circles
