# EVOLVE-BLOCK-START
import numpy as np
import time


def _project_entropy(P, r):
    """Compute Shannon entropy of projection x + r*y for distribution P[y,x]."""
    n = P.shape[0]
    # Build projection bins
    # For integer x,y in 0..n-1, x + r*y takes specific values
    # We'll pool by exact value
    vals = []
    probs = []
    for y in range(n):
        for x in range(n):
            v = x + r * y
            p = P[y, x]
            if p > 1e-15:
                # Find if this value exists
                found = False
                for i in range(len(vals)):
                    if vals[i] == v:
                        probs[i] += p
                        found = True
                        break
                if not found:
                    vals.append(v)
                    probs.append(p)
    # Compute entropy
    H = 0.0
    for p in probs:
        if p > 1e-15:
            H -= p * np.log2(p)
    return H


def _project_entropy_fast(P, r):
    """Faster entropy computation using numpy."""
    n = P.shape[0]
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[y,x] = x, Y[y,x] = y
    V = X + r * Y  # V[y,x] = x + r*y
    # Get unique values and sum probabilities
    unique_vals, inverse = np.unique(V, return_inverse=True)
    probs = np.zeros(len(unique_vals))
    np.add.at(probs, inverse, P)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _project_entropy_and_grad(P, r):
    """Return entropy H_r and gradient w.r.t. P."""
    n = P.shape[0]
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    V = X + r * Y
    unique_vals, inverse = np.unique(V, return_inverse=True)
    probs = np.zeros(len(unique_vals))
    np.add.at(probs, inverse, P)
    
    # Mask for positive probabilities
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(P)
    
    p_m = probs[mask]
    log_p = np.log2(p_m)
    
    # dH/dP[y,x] = -log2(prob_of_bin) / ln(2) ... wait
    # H = -sum_j p_j log2(p_j)
    # dH/dp_j = -log2(p_j) - 1/ln(2)
    # dH/dP[y,x] = dH/dp_{bin(y,x)} * dp_{bin}/dP[y,x] = dH/dp_{bin} * 1
    # So gradient = -(log2(p_bin) + 1/ln(2)) for each cell, where p_bin is the bin probability
    
    # Actually: dH/dP[y,x] = -log2(p_bin) - 1/ln(2) ... let me recompute
    # H = -sum_j p_j ln(p_j) / ln(2)
    # dH/dp_j = -(ln(p_j) + 1) / ln(2)
    # dH/dP[y,x] = dH/dp_{bin} * 1 = -(ln(p_bin) + 1) / ln(2)
    
    grad_bin = -(np.log(p_m) + 1) / np.log(2)
    
    # Map back to P space
    grad = np.zeros_like(P)
    # For each cell, its gradient is grad_bin[inverse[y,x]] if the bin has positive prob
    # But we need to handle the case where bin prob is zero
    full_grad_bin = np.zeros(len(unique_vals))
    full_grad_bin[mask] = grad_bin
    grad = full_grad_bin[inverse]
    
    H = -np.sum(p_m * log_p)
    return H, grad


def _soft_max(vals, temp=1.0):
    """Soft max approximation."""
    m = np.max(vals)
    return m + temp * np.log(np.sum(np.exp((vals - m) / temp)))


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit, keep safety margin
    
    # Initial: uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Precompute projection info
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    slopes_D = [-1.0]
    slopes_M = [0.0, 1.0, 2.0]
    
    V_D = X + slopes_D[0] * Y
    V_M = [X + r * Y for r in slopes_M]
    
    unique_D, inv_D = np.unique(V_D, return_inverse=True)
    unique_M = [np.unique(V)[0] for V in V_M]
    inv_M = [inv for _, inv in [(V, np.unique(V, return_inverse=True)[1]) for V in V_M]]
    
    def compute_objective(P):
        """Compute D/M and gradients."""
        # D = H_{-1}
        probs_D = np.zeros(len(unique_D))
        np.add.at(probs_D, inv_D, P)
        mask_D = probs_D > 1e-15
        if not np.any(mask_D):
            return 0.0, None, None
        p_D = probs_D[mask_D]
        D = -np.sum(p_D * np.log2(p_D))
        
        # M = max of H_r for r in {0,1,2}
        Hs_M = []
        grads_M = []
        for i, r in enumerate(slopes_M):
            probs_M = np.zeros(len(unique_M[i]))
            np.add.at(probs_M, inv_M[i], P)
            mask_M = probs_M > 1e-15
            if not np.any(mask_M):
                Hs_M.append(0.0)
                grads_M.append(np.zeros_like(P))
                continue
            p_M = probs_M[mask_M]
            H = -np.sum(p_M * np.log2(p_M))
            Hs_M.append(H)
            
            # Gradient of H_M[i]
            grad_bin = -(np.log(p_M) + 1) / np.log(2)
            full_grad = np.zeros(len(unique_M[i]))
            full_grad[mask_M] = grad_bin
            grads_M.append(full_grad[inv_M[i]])
        
        M = max(Hs_M)
        if M < 1e-9:
            return 0.0, None, None
        
        ratio = D / M
        
        # Gradient of ratio: d(D/M)/dP = (dD/dP * M - D * dM/dP) / M^2
        # dM/dP = d(max_i H_i)/dP = dH_k/dP where k = argmax
        # For simplicity, use the gradient of the max term
        k = np.argmax(Hs_M)
        dM_dP = grads_M[k]
        
        # Gradient of D
        grad_D_bin = -(np.log(p_D) + 1) / np.log(2)
        full_grad_D = np.zeros(len(unique_D))
        full_grad_D[mask_D] = grad_D_bin
        dD_dP = full_grad_D[inv_D]
        
        grad_ratio = (dD_dP * M - D * dM_dP) / (M * M)
        
        return ratio, grad_ratio, (D, M, Hs_M)
    
    # Gradient ascent with simplex projection
    best_P = P.copy()
    best_score, _, _ = compute_objective(P)
    
    lr = 0.01
    n_iter = 50000
    patience = 5000
    no_improve = 0
    
    for it in range(n_iter):
        if time.time() > deadline - 30:
            break
        
        score, grad, info = compute_objective(P)
        if grad is None:
            break
        
        if score > best_score + 1e-12:
            best_score = score
            best_P = P.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > patience:
            break
        
        # Projected gradient ascent: P_new = P + lr * grad, then project to simplex
        P_raw = P + lr * grad
        # Project to probability simplex: non-negative, sum to 1
        # Use iterative method
        P_new = P_raw.copy()
        # Simple projection: clip and normalize (not exact but works)
        P_new = np.maximum(P_new, 0)
        s = P_new.sum()
        if s < 1e-15:
            P_new = np.ones((n, n)) / (n * n)
        else:
            P_new = P_new / s
        
        # Check if we're stuck
        if np.allclose(P_new, P, atol=1e-15):
            # Try smaller learning rate or restart
            lr *= 0.5
            if lr < 1e-10:
                break
            continue
        
        P = P_new
    
    # Use the best found
    P = best_P
    
    # Final validation and normalization
    P = np.maximum(P, 0)
    s = P.sum()
    if s < 1e-15:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
    
    # Ensure finite
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.maximum(P, 0)
    s = P.sum()
    if s < 1e-15:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
    
    return P


# EVOLVE-BLOCK-END
