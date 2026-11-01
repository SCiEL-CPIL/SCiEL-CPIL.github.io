import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    idx_i, idx_j = np.triu_indices(N, k=1)
    num_pairs = len(idx_i)

    def get_distances(P):
        diff = P[idx_i] - P[idx_j]
        d = np.sum(diff**2, axis=1)
        return d, diff

    def objective_and_grad(P, alpha, reg_weight=1e-6):
        d, diff = get_distances(P)
        
        # Regularization to fix scale and center
        # Add small penalty for large coordinates
        # This doesn't change the optimal ratio much but helps convergence
        reg = reg_weight * np.sum(P**2)
        
        # Smoothed Max
        max_d = np.max(d)
        sum_exp_pos = np.sum(np.exp(alpha * (d - max_d)))
        M = (np.log(sum_exp_pos) + alpha * max_d) / alpha
        
        # Smoothed Min
        min_d = np.min(d)
        sum_exp_neg = np.sum(np.exp(-alpha * (d - min_d)))
        m = -(np.log(sum_exp_neg) - alpha * min_d) / alpha
        
        val = (M / m) ** 2 + reg
        
        if not np.isfinite(val):
            return val, np.zeros_like(P)
            
        # Gradients for ratio part
        w_pos = np.exp(alpha * (d - max_d)) / sum_exp_pos
        w_neg = np.exp(-alpha * (d - min_d)) / sum_exp_neg
        
        grad_M = np.zeros_like(P)
        grad_m = np.zeros_like(P)
        
        for k in range(num_pairs):
            i = idx_i[k]
            j = idx_j[k]
            diff_k = diff[k]
            
            contrib_M = w_pos[k]
            grad_M[i] -= 2.0 * contrib_M * diff_k
            grad_M[j] += 2.0 * contrib_M * diff_k
            
            contrib_m = w_neg[k]
            grad_m[i] -= 2.0 * contrib_m * diff_k
            grad_m[j] += 2.0 * contrib_m * diff_k
            
        factor = 2.0 * M / (m**3)
        grad_ratio = factor * (m * grad_M - M * grad_m)
        
        # Gradient for regularization
        grad_reg = 2.0 * reg_weight * P
        
        grad_total = grad_ratio + grad_reg
        
        return val, grad_total

    def run_optimization(P_init, alphas):
        P = P_init.copy()
        for alpha in alphas:
            def obj_wrapper(P):
                return objective_and_grad(P, alpha)
            
            # Check time before starting a new optimization phase
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    obj_wrapper, 
                    P, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                P = res.x
            except Exception:
                continue
        return P

    def score(P):
        d, _ = get_distances(P)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min < 1e-10:
            return np.inf
        return (d_max / d_min) ** 2

    best_P = None
    best_score = np.inf
    
    inits = []
    
    # 1. Random points
    np.random.seed(42)
    for _ in range(15):
        P_rand = np.random.uniform(-1, 1, (N, D))
        inits.append(P_rand)
        
    # 2. Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (2 * i) / (n - 1)
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / phi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    P_sphere = fibonacci_sphere(N)
    inits.append(P_sphere)
    
    # 3. Lattice
    grid_pts = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts, dtype=float)
    grid_pts -= grid_pts.mean(axis=0)
    # Scale to unit variance roughly
    grid_pts /= np.std(grid_pts)
    P_lattice = grid_pts[:14]
    inits.append(P_lattice)
    
    # 4. Two rings
    # 7 points on ring 1, 7 on ring 2
    angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r1 = 1.0
    r2 = 1.0
    z1 = 0.5
    z2 = -0.5
    P_rings = np.vstack([
        np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1), np.full(7, z1)]),
        np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2), np.full(7, z2)])
    ])
    inits.append(P_rings)

    alphas = [1.0, 5.0, 20.0, 100.0]
    
    for P_init in inits:
        if time.time() > deadline - 5:
            break
            
        try:
            P_opt = run_optimization(P_init, alphas)
            s = score(P_opt)
            if s < best_score:
                best_score = s
                best_P = P_opt
        except Exception:
            continue
            
    if best_P is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(np.meshgrid(np.arange(side), np.arange(side), np.arange(side)))
        grid = np.array(grid).reshape(3, -1).T
        best_P = grid[:N].astype(float)
        
    return best_P
