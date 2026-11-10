# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # Helper to compute smooth min/max
    def smooth_min_max_ratio(P, alpha=100.0):
        """
        Minimizes the smooth approximation of d_max / d_min.
        P: (n, 2) array of points.
        """
        D = pdist(P)
        if np.all(D == 0):
            return 1e10
        # Softmax and Softmin
        # To avoid overflow, subtract max/min
        D_max = np.max(D)
        D_min = np.min(D)
        
        # Softmax: (1/alpha) * log(sum(exp(alpha * D)))
        # Softmin: -(1/alpha) * log(sum(exp(-alpha * D)))
        
        # Normalized softmax/softmin for stability
        exp_D = np.exp(alpha * (D - D_max))
        softmax_val = (D_max + np.log(np.sum(exp_D)) / alpha)
        
        exp_neg_D = np.exp(-alpha * (D - D_min))
        softmin_val = (D_min - np.log(np.sum(exp_neg_D)) / alpha)
        
        if softmin_val < 1e-10:
            return 1e10
            
        return softmax_val / softmin_val

    def objective_flat(x_flat, alpha=100.0):
        P = x_flat.reshape(n, d)
        return smooth_min_max_ratio(P, alpha)

    # Initializations
    inits = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    P_grid = np.column_stack((X.ravel(), Y.ravel()))
    inits.append(P_grid)

    # 2. Hexagonal-like Grid (Staggered)
    # 4 rows, 4 cols
    # Row 0: 0, 1, 2, 3
    # Row 1: 0.5, 1.5, 2.5, 3.5 (shifted by 0.5)
    # Spacing in x: 1, Spacing in y: sqrt(3)/2 ~ 0.866
    P_hex = np.zeros((16, 2))
    for i in range(4):
        for j in range(4):
            P_hex[i*4+j, 0] = j + (i % 2) * 0.5
            P_hex[i*4+j, 1] = i * np.sqrt(3)/2
    # Center it
    P_hex -= np.mean(P_hex, axis=0)
    inits.append(P_hex)

    # 3. Circle with jitter (less likely to be optimal but good for diversity)
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    P_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(P_circle)

    # 4. Random symmetric
    np.random.seed(42)
    P_rand = np.random.randn(16, 2)
    P_rand -= np.mean(P_rand, axis=0)
    inits.append(P_rand)

    best_P = None
    best_score = -1e10
    best_ratio = 0

    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        # Try a few different alpha values for smoothness
        alphas = [50.0, 100.0, 200.0]
        
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            # Define objective with fixed alpha
            def obj(x):
                return objective_flat(x, alpha)
            
            x0 = init.flatten()
            
            # Use L-BFGS-B
            try:
                res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                if res.success or res.fun < 1e10:
                    P_opt = res.x.reshape(n, d)
                    # Calculate actual ratio
                    D = pdist(P_opt)
                    d_min = np.min(D)
                    d_max = np.max(D)
                    if d_max > 0:
                        ratio = (d_min / d_max) ** 2
                    else:
                        ratio = 0
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_P = P_opt
                        best_score = ratio
                        
            except Exception:
                pass

    # If no optimization succeeded or finished, return best initial or a default
    if best_P is None:
        # Fallback to grid
        best_P = inits[0]
        D = pdist(best_P)
        best_ratio = (np.min(D) / np.max(D)) ** 2

    # Final Polish: Use the best found P as start for a high-precision run
    if time.time() < deadline - 5:
        P_start = best_P.flatten()
        # Use higher alpha for sharper min/max
        alpha_final = 500.0
        def obj_final(x):
            return objective_flat(x, alpha_final)
        
        try:
            res_final = minimize(obj_final, P_start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res_final.fun < 1e10:
                P_final = res_final.x.reshape(n, d)
                D = pdist(P_final)
                d_min = np.min(D)
                d_max = np.max(D)
                if d_max > 0:
                    ratio_final = (d_min / d_max) ** 2
                    if ratio_final > best_ratio:
                        best_P = P_final
        except Exception:
            pass

    return best_P

# EVOLVE-BLOCK-END
