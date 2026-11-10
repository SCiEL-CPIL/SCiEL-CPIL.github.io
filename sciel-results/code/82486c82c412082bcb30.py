# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the distribution P on a 10x10 grid to maximize the ratio 
    H(x-y) / max(H(x), H(x+y), H(x+2y)).
    """
    n = 10
    # Precompute projection indices for each cell (y, x)
    # r = -1: x - y. Range -9 to 9. Offset by 9 -> 0 to 18.
    # r = 0: x. Range 0 to 9.
    # r = 1: x + y. Range 0 to 18.
    # r = 2: x + 2y. Range 0 to 27.
    
    ys, xs = np.indices((n, n))
    idx_m1 = xs - ys + 9  # Shape (10, 10), values 0..18
    idx_0 = xs            # Shape (10, 10), values 0..9
    idx_1 = xs + ys       # Shape (10, 10), values 0..18
    idx_2 = xs + 2 * ys   # Shape (10, 10), values 0..27
    
    bins_m1 = 19
    bins_0 = 10
    bins_1 = 19
    bins_2 = 28
    
    def project(P, idx, num_bins):
        # P is (10,10). idx is (10,10).
        # Returns array of size num_bins
        q = np.zeros(num_bins)
        np.add.at(q, idx.ravel(), P.ravel())
        return q

    def entropy(q):
        # Shannon entropy in bits
        # Ignore zeros
        mask = q > 1e-12
        if not np.any(mask):
            return 0.0
        q_m = q[mask]
        return -np.sum(q_m * np.log2(q_m))

    def entropy_grad(q, idx, num_bins):
        # Gradient of H(q) wrt P
        # dH/dP_{y,x} = -1/ln(2) * log2(q[idx[y,x]])
        # Note: q depends on P.
        # We compute the gradient for the full P matrix
        mask = q > 1e-12
        # If q is small, log is -inf. We clip q for stability in gradient
        q_safe = np.maximum(q, 1e-12)
        log_q = np.log2(q_safe)
        # grad_flat has shape (100,)
        grad_flat = -log_q[idx.ravel()] / np.log(2)
        return grad_flat.reshape((n, n))

    def objective_and_grad(P, beta=10.0):
        # Compute entropies
        q_m1 = project(P, idx_m1, bins_m1)
        q_0 = project(P, idx_0, bins_0)
        q_1 = project(P, idx_1, bins_1)
        q_2 = project(P, idx_2, bins_2)
        
        H_m1 = entropy(q_m1)
        H_0 = entropy(q_0)
        H_1 = entropy(q_1)
        H_2 = entropy(q_2)
        
        # Smooth max of H_0, H_1, H_2
        Hs = np.array([H_0, H_1, H_2])
        M = np.log(np.sum(np.exp(beta * Hs))) / beta
        
        # Gradients
        g_m1 = entropy_grad(q_m1, idx_m1, bins_m1)
        g_0 = entropy_grad(q_0, idx_0, bins_0)
        g_1 = entropy_grad(q_1, idx_1, bins_1)
        g_2 = entropy_grad(q_2, idx_2, bins_2)
        
        # Gradient of M wrt P
        # dM/dP = (1/beta) * sum_i exp(beta H_i) * dH_i/dP / sum_j exp(beta H_j)
        weights = np.exp(beta * Hs)
        weights /= np.sum(weights)
        
        g_M = weights[0] * g_0 + weights[1] * g_1 + weights[2] * g_2
        
        # Total gradient for maximizing H_m1 - M
        grad = g_m1 - g_M
        
        # Raw value for tracking
        raw_val = H_m1 / max(H_0, H_1, H_2) if max(H_0, H_1, H_2) > 1e-9 else 0.0
        
        return raw_val, grad

    def project_simplex(v):
        # Project vector v onto the simplex {x >= 0, sum(x) = 1}
        # v is flat array of length N
        N = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    # Optimization Loop
    start_time = time.time()
    time_limit = 590.0  # Safety margin
    
    best_P = None
    best_val = -np.inf
    
    # Multiple restarts
    n_restarts = 10
    steps_per_restart = 2000
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        if r == 0:
            P = np.ones((n, n)) / (n * n)
        elif r == 1:
            # Diagonal x+y = 9
            P = np.zeros((n, n))
            for y in range(n):
                x = n - 1 - y
                P[y, x] = 1.0 / n
        elif r == 2:
            # Diagonal x-y = 0
            P = np.zeros((n, n))
            for y in range(n):
                P[y, y] = 1.0 / n
        else:
            # Random sparse
            P = np.random.rand(n, n)
            P[P < 0.5] = 0
            P /= np.sum(P)
            
        P = P.copy()
        
        # Adaptive step size
        step = 0.1
        
        for s in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            val, grad = objective_and_grad(P)
            
            if val > best_val:
                best_val = val
                best_P = P.copy()
                
            # Gradient ascent
            P_new = P + step * grad
            
            # Project to simplex
            P_flat = project_simplex(P_new.ravel())
            P = P_flat.reshape((n, n))
            
            # Simple step size adjustment
            # If we moved too far, reduce step. 
            # Here we just keep it fixed or decay slowly.
            step *= 0.999
            
            # Early stopping if gradient is small
            if np.max(np.abs(grad)) < 1e-6:
                break
                
    # Final polish: ensure valid output
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
        
    # Clip and normalize
    best_P = np.maximum(best_P, 0)
    s = np.sum(best_P)
    if s < 1e-9:
        return np.ones((n, n)) / (n * n)
    best_P /= s
        
    return best_P

# EVOLVE-BLOCK-END
