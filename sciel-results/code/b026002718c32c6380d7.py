import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Objective: Maximize J(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Fix ||h||_1 = 1. Maximize ||h||_2^2 / ||h*h||_inf.
    # Minimize -J(h).
    
    def neg_obj_and_grad(h):
        h = np.asarray(h)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_val = np.max(conv)
        
        if max_val < 1e-15:
            return 0.0, np.zeros_like(h)
        
        obj_val = -(l2_sq / max_val)
        
        # Gradient calculation
        # d(l2_sq)/dh = 2h
        # d(max_val)/dh: subgradient. We use the average gradient over all indices achieving the max.
        # This makes the gradient continuous in practice (smoothing the max).
        
        # Find indices of max
        max_indices = np.where(conv == max_val)[0]
        
        grad_max = np.zeros(N)
        for k in max_indices:
            # conv[k] = sum_{i} h[i] * h[i+k] (with offset)
            # d(conv[k])/d(h[j]) = h[j+k] + h[j-k] (adjusted for bounds)
            # In numpy convolve 'full', index k corresponds to lag k - (N-1).
            # Let lag = k - (N - 1).
            lag = k - (N - 1)
            
            # Term 1: h[j] * h[j+lag] -> d/dh[j] = h[j+lag]
            # Term 2: h[j+lag] * h[j] -> d/dh[j] = h[j] (if we consider h[j+lag] as the varying term? No, product rule)
            # Let's derive carefully.
            # c_k = sum_i h_i h_{i+k}
            # d c_k / d h_j:
            # If j is in the first slot: h_{j+k}
            # If j is in the second slot: h_{j-k}
            
            # Indices for first slot: i=j => h_{j+k} must exist.
            # Indices for second slot: i+k=j => i=j-k => h_{j-k} must exist.
            
            # Vectorized:
            # grad1[j] = h[j+k] if 0 <= j+k < N else 0
            # grad2[j] = h[j-k] if 0 <= j-k < N else 0
            
            # Using np.roll or slicing is faster, but loop is fine for N=50.
            # We add contributions from all max lags.
            
            for j in range(N):
                val = 0.0
                idx1 = j + lag
                if 0 <= idx1 < N:
                    val += h[idx1]
                idx2 = j - lag
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad_max[j] += val
            
        grad_max /= len(max_indices)
        
        # Q = L2^2 / P
        # dQ/dh = (P * 2h - L2^2 * grad_P) / P^2
        grad_obj = (max_val * 2 * h - l2_sq * grad_max) / (max_val ** 2)
        
        # We are minimizing -Q, so gradient of -Q is -grad_obj
        grad_neg_obj = -grad_obj
        
        return obj_val, grad_neg_obj

    def neg_obj_only(h):
        val, _ = neg_obj_and_grad(h)
        return val

    best_h = None
    best_score = -np.inf
    
    # Restart with different initializations
    # 1. Uniform
    # 2. Sparse (delta)
    # 3. Random
    
    inits = []
    inits.append(np.ones(N) / N)
    
    # Sparse init: try a few random sparse vectors
    for _ in range(5):
        h_init = np.zeros(N)
        k = np.random.randint(1, 10)
        idx = np.random.choice(N, k, replace=False)
        h_init[idx] = 1.0 / k
        inits.append(h_init)
        
    # Random init
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init /= np.sum(h_init)
        inits.append(h_init)

    for h0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # SLSQP
            res = minimize(
                neg_obj_only, 
                h0, 
                method='SLSQP', 
                jac=neg_obj_and_grad,
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            h_res = np.clip(res.x, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res /= s
            
            score = _score_raw(h_res)
            if score > best_score:
                best_score = score
                best_h = h_res.copy()
                
        except Exception:
            pass

    # If no result, fallback to uniform
    if best_h is None:
        best_h = np.ones(N) / N
    else:
        # Final polish: Local search on best_h
        # Simple hill climbing
        h_curr = best_h.copy()
        curr_score = _score_raw(h_curr)
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try small perturbations
            for _ in range(20):
                h_test = h_curr.copy()
                # Perturb 2-3 elements
                idxs = np.random.choice(N, 3, replace=False)
                h_test[idxs] += np.random.randn(3) * 0.01
                h_test = np.clip(h_test, 0, None)
                s = np.sum(h_test)
                if s > 0:
                    h_test /= s
                s_test = _score_raw(h_test)
                if s_test > curr_score:
                    curr_score = s_test
                    h_curr = h_test
                    improved = True
            if not improved:
                break
        
        if curr_score > best_score:
            best_h = h_curr
            best_score = curr_score

    return list(best_h)
