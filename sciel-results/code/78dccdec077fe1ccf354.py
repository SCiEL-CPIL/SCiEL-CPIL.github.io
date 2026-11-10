import numpy as np
import time

def _project_to_simplex(v):
    """
    Project vector v onto the probability simplex (sum=1, v>=0).
    Uses the standard O(N log N) algorithm.
    """
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def _compute_P_and_grad(support, weights):
    """
    Compute P and gradient w.r.t weights for fixed support.
    """
    n = len(support)
    if n == 0:
        return 0.0, np.zeros(0)
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Ensure normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0, np.zeros(n)
    w = w / total_w
    
    # 1. Pair sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # 2. Thresholds T_kl = 2*s_l - s_k
    # Note: indices in P formula: sum_{k,l} w_k w_l F(2*s_l - s_k)
    # Here k is X3, l is X4.
    T = 2.0 * s[None, :] - s[:, None]
    
    # Flatten
    S_flat = S.flatten()
    T_flat = T.flatten()
    
    # Weights matrix W = w w^T
    W = w[:, None] * w[None, :]
    W_flat = W.flatten()
    
    # --- Compute F_kl ---
    # F_kl = sum_{i,j: S_ij < T_kl} W_ij
    # Sort S_flat
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    # For each T_kl, find count of S_ij < T_kl
    # searchsorted side='left' gives first index >= T_kl.
    # So indices < idx are < T_kl.
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Get cumulative sum up to idx-1
    # If idx=0, sum is 0.
    # cum_W_S is 0-indexed. sum of first idx elements is cum_W_S[idx-1] if idx>0.
    # We can use a helper array of zeros at the beginning to handle idx=0 easily?
    # Or just clip.
    # Let's create a padded cumsum: [0, cum_W_S[0], ...]
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    # --- Compute H_ij ---
    # H_ij = sum_{k,l: T_kl > S_ij} W_kl
    # Note: W_kl is the same matrix as W_ij (just different indices).
    # Sort T_flat
    order_T = np.argsort(T_flat, kind='stable')
    T_sorted = T_flat[order_T]
    W_sorted_T = W_flat[order_T]
    
    # We want sum of W where T > S_ij.
    # searchsorted side='right' gives first index > S_ij.
    idxs_H = np.searchsorted(T_sorted, S_flat, side='right')
    
    # Sum of W_sorted_T from idxs_H to end.
    # Suffix sum: suff[i] = sum(W_sorted_T[i:])
    # suff = cumsum(reverse(W))[::-1]
    rev_W = W_sorted_T[::-1]
    cum_rev_W = np.cumsum(rev_W)
    suff_W = cum_rev_W[::-1]
    
    # Handle case where idxs_H == n (sum is 0)
    padded_suff_W = np.concatenate((suff_W, [0.0]))
    H_flat = padded_suff_W[idxs_H]
    H = H_flat.reshape(n, n)
    
    # --- Compute P ---
    # P = sum_{k,l} w_k w_l F_kl
    P = np.sum(W * F)
    
    # --- Compute Gradient ---
    # G_m = w_m * ( sum_l F_ml + sum_k F_km + sum_j H_mj + sum_i H_im )
    # row_sums_F = F.sum(axis=1)
    # col_sums_F = F.sum(axis=0)
    # row_sums_H = H.sum(axis=1)
    # col_sums_H = H.sum(axis=0)
    
    term_F = F.sum(axis=1) + F.sum(axis=0)
    term_H = H.sum(axis=1) + H.sum(axis=0)
    
    grad = w * (term_F + term_H)
    
    return float(P), grad

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Grid sizes to try
    grid_sizes = [64, 128, 256, 512]
    
    # We will try a few random restarts for the weights on the best grid
    # Since the objective is not convex, multiple restarts help.
    
    for N in grid_sizes:
        if time.time() > deadline - 30.0:
            break
            
        # Create uniform grid on [0, 1]
        # Scale invariance means [0,1] is fine.
        support = np.linspace(0.0, 1.0, N)
        
        # Number of restarts depends on N
        # Larger N is more expensive per step, so fewer restarts or fewer steps?
        # Let's allow more steps for smaller N, fewer for larger?
        # Actually, O(N^2 log N) is fast.
        # Let's do 5 restarts for N=64, 3 for N=128, 2 for N=256, 1 for N=512.
        n_restarts = max(1, 10 - N // 32)
        
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize weights
            # Uniform or random Dirichlet?
            # Uniform is a good start. Random Dirichlet helps escape.
            if r == 0:
                w = np.ones(N) / N
            else:
                w = np.random.dirichlet(np.ones(N))
            
            # Projected Gradient Ascent
            # Step size: adaptive or fixed?
            # Let's use a fixed small step size and decay it.
            # Or use a simple line search?
            # For simplicity, fixed step size with decay.
            step_size = 0.05
            decay = 0.999
            max_steps = 2000 if N <= 128 else 1000
            
            w_curr = w.copy()
            P_curr = -1.0
            grad_curr = None
            
            for step in range(max_steps):
                if time.time() > deadline - 5.0:
                    break
                
                P_val, grad = _compute_P_and_grad(support, w_curr)
                
                if P_val > best_score:
                    best_score = P_val
                    best_support = support.copy()
                    best_weights = w_curr.copy()
                
                if P_val > P_curr:
                    P_curr = P_val
                else:
                    # If stuck, reduce step size
                    step_size *= 0.9
                
                # Update
                w_next = w_curr + step_size * grad
                w_next = _project_to_simplex(w_next)
                
                # Check if change is small
                if np.linalg.norm(w_next - w_curr) < 1e-8:
                    break
                    
                w_curr = w_next
                step_size *= decay

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Cleanup: remove zero weights to reduce n?
    # The task allows any n. Smaller n is faster for evaluator?
    # "The evaluation builds all n^2 pairwise sums, so very large n is slow."
    # So we should prune points with very small weights.
    
    if best_support is not None:
        # Prune weights < 1e-6
        mask = best_weights > 1e-6
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        
        # Renormalize
        total_w = best_weights.sum()
        if total_w > 0:
            best_weights = best_weights / total_w
        else:
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])
            
        # Ensure float64
        best_support = best_support.astype(np.float64)
        best_weights = best_weights.astype(np.float64)
        
        # If pruning resulted in empty or invalid, fallback
        if len(best_support) == 0 or np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])

    return best_support, best_weights
