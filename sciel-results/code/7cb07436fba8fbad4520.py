import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    n = 8
    
    def _log_score_and_grad(x_flat):
        """
        Computes the objective function H(P) and its gradient.
        H(P) = sum_{i<j} log(d_ij) - 28 * log(D)
        """
        x = x_flat.reshape(n, 2)
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)
        
        # Mask for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d_pairs = d[mask]
        i_idx, j_idx = np.where(mask)
        
        if np.any(d_pairs < 1e-12):
            return 1e100, np.zeros_like(x_flat)
            
        log_sum = np.sum(np.log(d_pairs))
        
        # Diameter
        D = np.max(d_pairs)
        # Find indices of max distance
        # We need to be careful if there are multiple max distances.
        # For gradient, we can just pick one or average. Picking one is standard.
        max_idx = np.argmax(d_pairs)
        a, b = i_idx[max_idx], j_idx[max_idx]
        
        log_D = np.log(D)
        score_val = log_sum - 28.0 * log_D
        
        # Gradient
        grad = np.zeros((n, 2))
        
        # Gradient of sum log(d_ij)
        # For each point k, sum over j != k of (x_k - x_j) / d_kj^2
        # We can compute this using matrix operations or loops.
        # Since n=8, a loop is fine and less prone to index errors in vectorization.
        for k in range(n):
            for j in range(n):
                if k == j: continue
                d_kj = d[k, j]
                if d_kj < 1e-12: continue
                grad[k] += (x[k] - x[j]) / (d_kj * d_kj)
                
        # Gradient of -28 log(D)
        # Only non-zero for a and b
        # d/dx_a log(D) = (x_a - x_b) / D^2
        # d/dx_b log(D) = (x_b - x_a) / D^2
        if D > 1e-12:
            grad[a] -= 28.0 * (x[a] - x[b]) / (D * D)
            grad[b] -= 28.0 * (x[b] - x[a]) / (D * D)
            
        return -score_val, -grad.flatten() # Minimizing negative score

    def _score_from_pts(x_flat):
        x = x_flat.reshape(n, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d_pairs = d[mask]
        if np.any(d_pairs < 1e-12):
            return 0.0
        log_sum = np.sum(np.log(d_pairs))
        D = np.max(d_pairs)
        log_score = log_sum - 28.0 * np.log(D) + 16.0 * np.log(2) # + 32/2 log(2)
        return np.exp(log_score)

    best_score = 0.0
    best_x = None
    
    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x_oct = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    inits.append(x_oct)
    
    # 2. Perturbed Octagons
    rng = np.random.default_rng(42)
    for _ in range(20):
        noise = rng.normal(0, 0.2, size=x_oct.shape)
        inits.append(x_oct + noise)
        
    # 3. Two Concentric Squares
    # Square 1: radius r1, Square 2: radius r2
    for r1 in [0.5, 0.7, 1.0]:
        for r2 in [0.5, 0.7, 1.0]:
            for rot in [0, np.pi/4]:
                angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot
                angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot + np.pi/4
                pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
                pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
                x_init = np.vstack([pts1, pts2]).flatten()
                inits.append(x_init)
                
    # 4. Random Points
    for _ in range(10):
        x_rand = rng.uniform(-1, 1, size=(n, 2)).flatten()
        inits.append(x_rand)
        
    # Optimization Loop
    for x_init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _log_score_and_grad, 
                x_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 3]: # 1: max iter, 2: precision, 3: max cor
                # Check if valid
                x_opt = res.x
                # Ensure distinctness
                d_opt = _score_from_pts(x_opt)
                if d_opt > best_score:
                    best_score = d_opt
                    best_x = x_opt
                    
        except Exception:
            pass

    if best_x is None:
        return [[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)]
        
    # Scale best_x to diameter 2
    x_best = best_x.reshape(n, 2)
    diff = x_best[:, np.newaxis, :] - x_best[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    
    if D < 1e-9:
         return [[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)]
         
    x_scaled = x_best / (D/2.0)
    
    # Final check for distinctness
    d_scaled = np.sqrt(np.sum((x_scaled[:, np.newaxis, :] - x_scaled[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(d_scaled, 1.0)
    if np.min(d_scaled[d_scaled != 1.0]) < 1e-6:
        return [[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)]
        
    return x_scaled.tolist()
