import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Precompute projection values for each slope
    # P[y, x]
    # x in 0..9, y in 0..9
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy') # X[y, x], Y[y, x]
    # Wait, meshgrid with indexing='xy' returns X, Y of shape (ny, nx)
    # X[i, j] = x_j, Y[i, j] = y_i
    # So X[y, x] is x, Y[y, x] is y.
    
    # Slopes: -1, 0, 1, 2
    # Values:
    V_minus1 = X - Y  # x - y
    V_0 = X           # x
    V_1 = X + Y       # x + y
    V_2 = X + 2*Y     # x + 2y
    
    # Map values to bin indices to compute entropies efficiently
    # We can just use np.unique with return_inverse or bincount
    
    def compute_entropies_and_grads(P):
        # P is 10x10
        # Compute entropies
        # H_r = - sum Q log Q
        
        # For each slope, get the distribution Q
        # We can use np.bincount on the flattened values with weights P.flatten()
        
        # V_minus1 range: -9 to 9
        # V_0 range: 0 to 9
        # V_1 range: 0 to 18
        # V_2 range: 0 to 27
        
        # Helper to compute entropy and gradient for a projection
        def proj_entropy_grad(V, P):
            # Flatten
            v_flat = V.flatten()
            p_flat = P.flatten()
            
            # Get unique values and inverse indices
            # Since ranges are small, we can use bincount
            # But V can be negative. Shift it.
            min_v = np.min(v_flat)
            max_v = np.max(v_flat)
            shifted = v_flat - min_v
            
            # Bin counts
            Q = np.bincount(shifted, weights=p_flat, minlength=max_v - min_v + 1)
            
            # Entropy
            # Ignore zeros
            mask = Q > 1e-15
            if np.sum(mask) == 0:
                return 0.0, np.zeros_like(P)
            
            Q_norm = Q[mask]
            H = -np.sum(Q_norm * np.log2(Q_norm))
            
            # Gradient
            # dH/dP_i = -log2(Q[v_i]) - 1/ln(2)
            # Q[v_i] is the bin count for the bin containing v_i
            # We need to map each cell back to its bin count
            # Q_shifted[shifted[i]]
            
            # Create an array of Q values for each cell
            Q_cell = Q[shifted]
            
            # Avoid log(0)
            Q_cell_safe = np.where(Q_cell > 1e-15, Q_cell, 1.0)
            
            grad_flat = -np.log2(Q_cell_safe) - 1.0/np.log(2)
            
            # If Q_cell was 0, the derivative is technically undefined or infinite, 
            # but in practice if P_i > 0 and Q=0, that's impossible.
            # If P_i = 0, we don't care about the gradient direction as much, 
            # but we should set it to 0 to avoid issues?
            # Actually, if P_i = 0, it doesn't contribute to Q.
            # The formula holds.
            
            return H, grad_flat.reshape(P.shape)

        H_m1, G_m1 = proj_entropy_grad(V_minus1, P)
        H_0, G_0 = proj_entropy_grad(V_0, P)
        H_1, G_1 = proj_entropy_grad(V_1, P)
        H_2, G_2 = proj_entropy_grad(V_2, P)
        
        return H_m1, H_0, H_1, H_2, G_m1, G_0, G_1, G_2

    # Initialize P
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Optimization loop
    # Use gradient ascent on D - M
    # Or D / M
    # Let's use D - M for stability, then check ratio at the end.
    # Actually, maximizing D/M is equivalent to maximizing D - lambda M for some lambda? No.
    # Let's just maximize D - M. If D and M are close, this might be biased.
    # Let's try to maximize log(D/M) = log(D) - log(M).
    
    start_time = time.time()
    timeout = 590  # Leave 10s margin
    
    # Learning rate
    lr = 0.05
    # Decay learning rate
    lr_decay = 0.999
    
    best_P = P.copy()
    best_score = -np.inf
    
    for step in range(50000):
        if time.time() - start_time > timeout:
            break
            
        H_m1, H_0, H_1, H_2, G_m1, G_0, G_1, G_2 = compute_entropies_and_grads(P)
        
        # M = max(H_0, H_1, H_2)
        Hs = np.array([H_0, H_1, H_2])
        Gs = [G_0, G_1, G_2]
        
        max_idx = np.argmax(Hs)
        M = Hs[max_idx]
        G_M = Gs[max_idx]
        
        # Objective: D / M
        if M < 1e-9:
            score = 0
            grad_obj = G_m1
        else:
            score = H_m1 / M
            # d(D/M)/dP = (M * dD - D * dM) / M^2
            # We can just use (G_m1 - (H_m1/M) * G_M) as an approximation for direction?
            # Or simply G_m1 - G_M if we want to maximize D-M.
            # Let's try maximizing D - M first, it's simpler and robust.
            # If D-M is maximized, D/M is likely high.
            
        # Let's stick to D - M for the gradient step
        # It avoids division by zero issues and is smooth.
        grad_obj = G_m1 - G_M
        
        # Update
        P_new = P + lr * grad_obj
        
        # Project to simplex: clip and normalize
        P_new = np.clip(P_new, 0, None)
        s = np.sum(P_new)
        if s > 1e-15:
            P_new = P_new / s
        else:
            P_new = np.ones((n, n)) / (n*n)
            
        P = P_new
        
        # Evaluate actual ratio
        H_m1_curr, H_0_curr, H_1_curr, H_2_curr, _, _, _, _ = compute_entropies_and_grads(P)
        M_curr = max(H_0_curr, H_1_curr, H_2_curr)
        if M_curr > 1e-9:
            curr_score = H_m1_curr / M_curr
            if curr_score > best_score:
                best_score = curr_score
                best_P = P.copy()
        
        lr *= lr_decay
        
        # If lr is too small, break
        if lr < 1e-6:
            break
            
    return best_P
