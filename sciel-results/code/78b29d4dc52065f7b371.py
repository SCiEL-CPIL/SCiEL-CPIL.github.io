import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_val = -np.inf
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    best_H = 1.0

    # Number of random starts
    n_starts = 10
    
    for k in range(n_starts):
        # Random initialization
        np.random.seed(42 + k)
        # Scatter in 1x1 square, keep away from edges
        x = np.random.rand(N) * 0.6 + 0.2
        y = np.random.rand(N) * 0.6 + 0.2
        r = np.full(N, 0.05)
        W = 1.0
        H = 1.0
        
        v0 = np.concatenate([x, y, r, [W, H]])
        
        def penalty_objective(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            W = v[3*N]
            H = v[3*N+1]
            
            # Ensure positive radii and dims for calculation
            r_safe = np.maximum(r, 0.001)
            W_safe = max(W, 0.001)
            H_safe = max(H, 0.001)
            
            # Objective: Maximize sum of radii
            score = -np.sum(r_safe)
            
            # Penalty weight
            lam = 100.0
            
            # Overlap penalty
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    rad_sum = r_safe[i] + r_safe[j]
                    if dist < rad_sum:
                        overlap_pen += (rad_sum - dist)**2
            
            # Boundary penalty
            bound_pen = 0.0
            for i in range(N):
                # x bounds
                if x[i] < r_safe[i]:
                    bound_pen += (r_safe[i] - x[i])**2
                if x[i] + r_safe[i] > W_safe:
                    bound_pen += (x[i] + r_safe[i] - W_safe)**2
                # y bounds
                if y[i] < r_safe[i]:
                    bound_pen += (r_safe[i] - y[i])**2
                if y[i] + r_safe[i] > H_safe:
                    bound_pen += (y[i] + r_safe[i] - H_safe)**2
            
            # Perimeter penalty
            perim_pen = (2.0 * (W_safe + H_safe) - 4.0)**2
            
            return score + lam * (overlap_pen + bound_pen + perim_pen)

        res = minimize(penalty_objective, v0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.success:
            x_f = res.x[:N]
            y_f = res.x[N:2*N]
            r_f = res.x[2*N:3*N]
            W_f = res.x[3*N]
            H_f = res.x[3*N+1]
            
            # Verify and adjust
            # Check if valid
            valid = True
            if W_f <= 0 or H_f <= 0:
                valid = False
            else:
                for i in range(N):
                    if r_f[i] <= 0:
                        valid = False
                        break
                    if x_f[i] - r_f[i] < -1e-6 or x_f[i] + r_f[i] > W_f + 1e-6:
                        valid = False
                        break
                    if y_f[i] - r_f[i] < -1e-6 or y_f[i] + r_f[i] > H_f + 1e-6:
                        valid = False
                        break
                    for j in range(i+1, N):
                        dx = x_f[i] - x_f[j]
                        dy = y_f[i] - y_f[j]
                        dist_sq = dx*dx + dy*dy
                        rad_sum = r_f[i] + r_f[j]
                        if dist_sq < rad_sum**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r_f)
                if val > best_val:
                    best_val = val
                    best_x = x_f.copy()
                    best_y = y_f.copy()
                    best_r = r_f.copy()
                    best_W = W_f
                    best_H = H_f

    if best_x is None:
        # Fallback to a simple grid
        x = np.array([(i + 0.5) / 5 for i in range(5) for _ in range(5)][:N])
        y = np.array([(j + 0.5) / 5 for _ in range(5) for j in range(5)][:N])
        r = np.full(N, 0.09)
        W = 1.0
        H = 1.0
        return np.column_stack((x, y, r))

    # Scale to ensure perimeter is exactly 4 (handle floating point errors)
    perim = 2 * (best_W + best_H)
    if abs(perim - 4.0) > 1e-5:
        scale = 4.0 / perim
        best_x *= scale
        best_y *= scale
        best_r *= scale
        best_W *= scale
        best_H *= scale

    return np.column_stack((best_x, best_y, best_r))
