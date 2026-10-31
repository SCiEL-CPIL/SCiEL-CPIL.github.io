import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initialize h
    h = np.full(N, 0.5)
    
    # Parameters
    lr = 0.01
    lam = 10.0  # Sharpness of soft-max
    
    best_h = h.copy()
    best_score = np.inf
    
    # Precompute indices for lags
    # We only need to check a subset of lags if N is large, but N=100 is small.
    # Check all lags 0 to N-1.
    
    def compute_score(h):
        # h is 1D array
        scores = np.zeros(N)
        for k in range(N):
            # S_k = sum(h[0:N-k])
            # C_k = sum(h[0:N-k] * h[k:N])
            if k == 0:
                S_k = np.sum(h)
                C_k = np.sum(h**2)
            else:
                S_k = np.sum(h[:N-k])
                C_k = np.sum(h[:N-k] * h[k:])
            scores[k] = S_k - C_k
        return np.max(scores)
    
    current_score = compute_score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
        
    for iter in range(1000):
        if time.time() > deadline:
            break
            
        # Compute gradient of soft-max objective
        # J = log(sum(exp(lam * f_k)))
        # dJ/dh_i = lam * sum_k (exp(lam * f_k) / sum_j exp(lam * f_j)) * df_k/dh_i
        
        f = np.zeros(N)
        for k in range(N):
            if k == 0:
                f[k] = np.sum(h) - np.sum(h**2)
            else:
                f[k] = np.sum(h[:N-k]) - np.sum(h[:N-k] * h[k:])
        
        # Soft max weights
        max_f = np.max(f)
        exp_f = np.exp(lam * (f - max_f))
        weights = exp_f / np.sum(exp_f)
        
        # Gradient of f_k w.r.t h_i
        grad_h = np.zeros(N)
        for k in range(N):
            w = weights[k]
            if w < 1e-6:
                continue
            if k == 0:
                # f_0 = sum(h) - sum(h^2)
                # df_0/dh_i = 1 - 2h_i
                grad_h += w * (1 - 2 * h)
            else:
                # f_k = sum(h[:N-k]) - sum(h[:N-k] * h[k:])
                # d/dh_i sum(h[:N-k]) is 1 if i < N-k else 0
                # d/dh_i sum(h[:N-k] * h[k:]) 
                # Term h_i * h_{i+k} exists if i < N-k. Deriv: h_{i+k}
                # Term h_{i-k} * h_i exists if i-k >= 0 => i >= k. Deriv: h_{i-k}
                
                mask1 = np.arange(N) < (N - k)
                grad_h[mask1] += w * 1.0
                
                # Subtract derivative of C_k
                # h_i * h_{i+k}
                mask2 = np.arange(N) < (N - k)
                grad_h[mask2] -= w * h[k:][mask2] # h[k:N] aligned with h[0:N-k]
                
                # h_{i-k} * h_i
                mask3 = np.arange(N) >= k
                # h[i-k] for i in k..N-1
                # This corresponds to h[0:N-k]
                grad_h[mask3] -= w * h[:N-k][mask3-k]
                
        # Update h
        h -= lr * grad_h
        
        # Project to constraints: sum=0.5N, 0<=h<=1
        # Simple projection: clip and adjust sum
        h = np.clip(h, 0, 1)
        # Adjust sum
        diff = N * 0.5 - np.sum(h)
        if diff != 0:
            # Add diff/N to all, then clip again (might violate sum slightly)
            h += diff / N
            h = np.clip(h, 0, 1)
            # If still off, adjust largest/smallest
            # For simplicity, iterate projection a few times
            for _ in range(10):
                diff = N * 0.5 - np.sum(h)
                if abs(diff) < 1e-6:
                    break
                if diff > 0:
                    # Need to increase sum. Increase elements with most room (h < 1)
                    room = 1 - h
                    idx = np.argsort(room)[::-1]
                    for i in idx:
                        if diff <= 0: break
                        add = min(diff, room[i])
                        h[i] += add
                        diff -= add
                else:
                    # Need to decrease sum. Decrease elements with most value (h > 0)
                    val = h
                    idx = np.argsort(val)[::-1]
                    for i in idx:
                        if diff >= 0: break
                        sub = min(-diff, val[i])
                        h[i] -= sub
                        diff += sub
                        
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h
# EVOLVE-BLOCK-END
