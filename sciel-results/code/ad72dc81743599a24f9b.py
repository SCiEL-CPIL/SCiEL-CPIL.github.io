import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Initial guess for radius
    r_low = 0.05
    r_high = 0.08
    
    best_c = None
    best_r = 0.0
    
    # Helper to calculate energy and gradient
    def energy_and_grad(c_flat, r):
        c = c_flat.reshape(N, 2)
        # Distances
        # d_ij
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        # dist to walls: 0.5 - |x|, 0.5 - |y|
        bx = 0.5 - np.abs(c[:, 0])
        by = 0.5 - np.abs(c[:, 1])
        b = np.minimum(bx, by)
        
        # Energy
        # Overlap penalty
        viol_d = np.maximum(0, 2*r - d)
        E_d = np.sum(viol_d**2)
        
        # Boundary penalty
        viol_b = np.maximum(0, 2*r - b)
        E_b = np.sum(viol_b**2)
        
        E = E_d + E_b
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Pairwise gradient
        # d/dc_i sum max(0, 2r - d_ij)^2
        # = sum 2 * max(0, 2r - d_ij) * (-1) * (c_i - c_j)/d_ij
        # Note: d_ij is symmetric.
        
        # We compute for each i, sum over j
        # viol_d is NxN matrix
        # factor_ij = 2 * viol_d_ij / d_ij if viol_d_ij > 0 else 0
        # grad_i = - sum_j factor_ij * (c_i - c_j)
        
        # Efficient calculation:
        # Create matrix W where W_ij = 2 * max(0, 2r - d_ij) / d_ij
        W = np.zeros((N, N))
        mask = viol_d > 0
        W[mask] = 2 * viol_d[mask] / d[mask]
        
        # grad_i = - sum_j W_ij (c_i - c_j) = - (sum_j W_ij) c_i + sum_j W_ij c_j
        # row_sums = W.sum(axis=1)
        # grad = -row_sums[:, np.newaxis] * c + W @ c
        
        row_sums = W.sum(axis=1)
        grad_d = -row_sums[:, np.newaxis] * c + W.dot(c)
        
        # Boundary gradient
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is non-smooth. 
        # Approximate: if x is limiting, grad_x = -sign(x), grad_y = 0
        # if y is limiting, grad_x = 0, grad_y = -sign(y)
        # if tie, average?
        
        # Let's just use subgradient or simple penalty gradient.
        # viol_b > 0 implies b < 2r.
        # grad_b_i = -2 * viol_b_i * grad_b_i
        
        grad_b = np.zeros_like(c)
        for i in range(N):
            if viol_b[i] > 0:
                # Determine which constraint is active
                # If bx < by, active is x
                # If by < bx, active is y
                # If equal, both
                if bx[i] < by[i] - 1e-9:
                    grad_b[i, 0] = -np.sign(c[i, 0])
                elif by[i] < bx[i] - 1e-9:
                    grad_b[i, 1] = -np.sign(c[i, 1])
                else:
                    grad_b[i, 0] = -0.5 * np.sign(c[i, 0])
                    grad_b[i, 1] = -0.5 * np.sign(c[i, 1])
                grad_b[i] *= 2 * viol_b[i]
                
        grad = grad_d + grad_b
        
        return E, grad.flatten()

    def try_radius(r, max_restarts=3):
        best_E = np.inf
        best_c_local = None
        
        for k in range(max_restarts):
            # Init: Jittered grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            c_init = grid[:N].copy()
            # Add noise
            c_init += np.random.normal(0, 0.01, c_init.shape)
            # Clip to bounds
            c_init = np.clip(c_init, -0.5, 0.5)
            
            res = minimize(
                energy_and_grad,
                c_init.flatten(),
                args=(r,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'gtol': 1e-5}
            )
            
            if res.fun < best_E:
                best_E = res.fun
                best_c_local = res.x.reshape(N, 2)
                
            # Check time
            if time.time() - start_time > timeout - 5:
                break
                
        return best_E, best_c_local

    # Bisect
    # We want to find max r such that energy is close to 0.
    # Threshold for "feasible"
    eps = 1e-4
    
    for _ in range(10): # Max bisection steps
        if time.time() - start_time > timeout - 10:
            break
            
        r_mid = (r_low + r_high) / 2
        E, c_found = try_radius(r_mid)
        
        if E < eps:
            r_low = r_mid
            if c_found is not None:
                # Verify actual radius
                d = np.linalg.norm(c_found[:, np.newaxis, :] - c_found[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                min_d = np.min(d)
                bx = 0.5 - np.abs(c_found[:, 0])
                by = 0.5 - np.abs(c_found[:, 1])
                min_b = np.min(np.minimum(bx, by))
                actual_r = min(min_d / 2, min_b)
                
                if actual_r > best_r:
                    best_r = actual_r
                    best_c = c_found
        else:
            r_high = r_mid
            
    # If best_c is still None, use a grid
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c
