import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """Compute overlaps for all shifts. Returns array of length 2N (N pos, N neg shifts)."""
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _project(z):
    """Project z onto {h in [0,1]^N : sum(h) = N/2} using bisection on threshold."""
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _soft_max_value_grad(h, tau):
    """Compute soft-max value and gradient for the overlap objective."""
    O = _compute_O(h)
    Omax = float(O.max())
    # Center to prevent overflow in exp
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    # Precompute index matrices for gradient
    K = np.arange(N)[:, None]  # shift k
    I = np.arange(N)[None, :]  # position i
    
    # For positive shift k: O_pos[k] = sum_{j=k}^{N-1} SCALE * (h[j] - h[j]*h[j-k])
    # d O_pos[k] / d h[i] = SCALE * (1_{i>=k} - h[i-k]*1_{i>=k} - h[i]*1_{i+k<N})
    # = SCALE * (1_{i>=k} - h[i-k]*1_{i>=k} - h[i]*1_{i-(N-k)>...})
    # Let's be more careful.
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
    # = SCALE * (sum_{i=0}^{N-1} h[i] - sum_{i=0}^{k-1} h[i] - sum_{j=k}^{N-1} h[j]h[j-k])
    # = SCALE * (sum_{i=k}^{N-1} h[i] - sum_{j=k}^{N-1} h[j]h[j-k])
    # d/dh[i]:
    # if i >= k: SCALE * (1 - h[i-k])
    # if i < k: 0 from first term. Second term: sum_{j=k}^{N-1} h[j]h[j-k].
    #   If i = j-k for some j in [k, N-1], i.e., j = i+k in [k, N-1] => i in [0, N-1-k]
    #   Then derivative is -h[j] = -h[i+k]
    # So if i < k and i <= N-1-k: -SCALE * h[i+k]
    # if i < k and i > N-1-k: 0 (which means i+k >= N, impossible since i<N)
    
    # Similarly for negative shift:
    # O_neg[k] = SCALE * (sum_{i=0}^{N-1-k} h[i] - sum_{i=0}^{N-1-k} h[i]h[i+k])
    # d/dh[i]:
    # if i <= N-1-k: SCALE * (1 - h[i+k])
    # if i > N-1-k: 0 from first term. Second term:
    #   If i = i'+k for some i' in [0, N-1-k], i.e., i' = i-k in [0, N-1-k] => i in [k, N-1]
    #   Then derivative is -h[i'] = -h[i-k]
    # So if i > N-1-k and i >= k: -SCALE * h[i-k]
    
    ge_pos = (I >= K).astype(float)  # i >= k
    lt_neg_pos = (I < (N - K)).astype(float)  # i < N-k => i+k < N
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    ge_neg = (I <= (N - 1 - K)).astype(float)  # i <= N-1-k
    lt_neg_neg = (I > (N - 1 - K)).astype(float)  # i > N-1-k
    # Actually let's reuse similar masks
    
    # Gradient for O_pos w.r.t h
    Gpos = SCALE * (ge_pos * (1.0 - h[idx_p]) - (1.0 - ge_pos) * lt_neg_pos * h[idx_m])
    # Wait, the condition for the second term is i < k AND i + k < N.
    # i < k is (1 - ge_pos). i + k < N is (I < N - K).
    # So yes, (1 - ge_pos) * (I < N - K).
    
    # Gradient for O_neg w.r.t h
    # if i <= N-1-k: SCALE * (1 - h[i+k])
    # if i > N-1-k AND i >= k: -SCALE * h[i-k]
    ge_neg_mask = (I <= (N - 1 - K)).astype(float)
    lt_neg_mask = (I > (N - 1 - K)).astype(float)
    ge_k_mask = (I >= K).astype(float)
    
    Gneg = SCALE * (ge_neg_mask * (1.0 - h[idx_m]) - lt_neg_mask * ge_k_mask * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad

def _fista(h_init, deadline):
    h = _project(h_init.copy())
    h_prev = h.copy()
    t = 1.0
    L = 5.0  # Lipschitz estimate
    
    for i in range(5000):
        if time.time() > deadline - 1.0:
            break
        
        Omax, grad = _soft_max_value_grad(h, 80.0)
        
        step = 1.0 / L
        y = h - step * grad
        y_proj = _project(y)
        
        # Simple backtracking: just check if objective decreased
        O_new = float(_compute_O(y_proj).max())
        if O_new <= Omax + 1e-12:
            h = y_proj
            L = max(1.0, L / 1.2)
        else:
            L = L * 1.5
            if L > 1e6:
                break
            continue
        
        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        h_prev = h.copy()
        h = _project(h + ((t - 1) / t_new) * (h - h_prev))
        t = t_new
        
    return h

def _local_search_bb(h, deadline, max_iters=5000):
    """Bang-bang local search: swap a 1 and a 0 to reduce max overlap."""
    h_bb = (h >= 0.5).astype(float)
    diff = N / 2.0 - h_bb.sum()
    if diff > 0:
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            h_bb[zeros[:int(diff)]] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            h_bb[ones[:int(-diff)]] = 0.0
    
    best_val = float(_compute_O(h_bb).max())
    ones = np.where(h_bb == 1)[0]
    zeros = np.where(h_bb == 0)[0]
    
    for _ in range(max_iters):
        if time.time() > deadline - 1.0:
            break
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = float(_compute_O(h_test).max())
        if val < best_val - 1e-12:
            h_bb = h_test
            best_val = val
            ones = np.where(h_bb == 1)[0]
            zeros = np.where(h_bb == 0)[0]
        # Occasionally also try swapping a 1 with another 1? No, must keep sum.
        # Or try flipping a single bit and then fixing sum? That's more complex.
        
    return h_bb, best_val

def construct():
    start = time.time()
    deadline = start + 135.0  # Safety margin for 150s limit
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())
    
    idx = np.arange(N)
    inits = [seed.copy()]
    for f in (1, 2, 3, 5, 8, 13):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
        inits.append(0.5 + 0.25 * np.sin(2.0 * np.pi * f * idx / N))
    
    rng = np.random.default_rng(42)
    for _ in range(4):
        inits.append(rng.random(N))
    
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        try:
            # 1. Continuous optimization via FISTA
            h_cont = _fista(init, deadline)
            val_cont = float(_compute_O(h_cont).max())
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont
            
            # 2. Bang-bang local search from continuous solution
            h_bb, val_bb = _local_search_bb(h_cont, deadline, 4000)
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
            
            # 3. Try direct bang-bang from init
            h_bb2, val_bb2 = _local_search_bb(init, deadline, 3000)
            if val_bb2 < best_val:
                best_h, best_val = h_bb2, val_bb2
                
        except Exception:
            continue
            
    return best_h
