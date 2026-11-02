import numpy as np
import time

N = 100

def _overlaps(h):
    # h: array of length N in [0,1]
    # overlap(k) = sum_{i=0}^{N-k-1} h[i]*(1-h[i+k]) for k=0..N-1
    # return array of length N
    n = len(h)
    ov = np.empty(n)
    for k in range(n):
        m = n - k
        if m > 0:
            ov[k] = np.sum(h[:m] * (1.0 - h[k:k+m]))
        else:
            ov[k] = 0.0
    return ov

def _grad_overlaps(h):
    # Compute gradient of each overlap O_k w.r.t h
    # Returns a matrix G of shape (N, N) where G[k, j] = d O_k / d h_j
    n = len(h)
    G = np.zeros((n, n))
    for k in range(n):
        m = n - k
        if m <= 0:
            continue
        # For j in 0..m-1, h_j is the first factor in term i=j
        # d/dh_j (h_j (1-h_{j+k})) = 1 - h_{j+k}
        # Valid for j such that j+k < n => j < n-k = m
        if m > 0:
            G[k, :m] += (1.0 - h[k:k+m])
        
        # For j in k..n-1, h_j is the second factor in term i=j-k
        # Term is h_{j-k} (1-h_j)
        # d/dh_j (h_{j-k} (1-h_j)) = -h_{j-k}
        # Valid for j such that j-k >= 0 => j >= k
        if k < n:
            G[k, k:] += -h[:n-k]
    return G

def _project(h, target_sum):
    # Project h onto [0,1]^N with sum = target_sum
    # h is already clipped to [0,1] in previous steps? 
    # We use bisection on lambda
    # h_proj = clip(h - lambda, 0, 1)
    # Find lambda such that sum(h_proj) = target_sum
    
    lo = h.min() - 1.0
    hi = h.max() + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h_proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    # Adjust for any tiny numerical error
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-9:
        # Distribute difference among elements not at bounds
        mask = (h_proj > 0.001) & (h_proj < 0.999)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            # Re-clip just in case
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Initial guess: constant 0.5
    h = np.full(N, 0.5)
    
    # Optional: Try a few random initializations or a ramp to escape local minima?
    # The problem is convex? No, it's min-max of bilinear forms. 
    # However, for N=100, gradient descent often finds good solutions.
    
    # To improve robustness, let's try a few restarts if time permits.
    best_h = h.copy()
    best_val = np.max(_overlaps(h))
    
    # Learning rate for projected gradient descent
    lr = 0.1 
    # We can use a simple line search or fixed step with decay
    
    # Number of iterations
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Compute overlaps
        ov = _overlaps(h)
        
        # Compute gradient of LogSumExp(ov)
        # To avoid overflow, subtract max
        max_ov = np.max(ov)
        exp_ov = np.exp(ov - max_ov)
        Z = np.sum(exp_ov)
        
        # Gradient of LogSumExp: (1/Z) * sum(exp(ov_k) * grad_ov_k)
        # We need grad_ov_k for all k
        # This is expensive to compute full matrix G every time.
        # Let's compute the gradient vector directly.
        # grad_h[j] = sum_k (exp(ov_k)/Z) * (d ov_k / d h_j)
        
        # d ov_k / d h_j = I[j < N-k](1 - h_{j+k}) - I[j >= k] h_{j-k}
        
        # We can compute this efficiently.
        # Let w_k = exp(ov_k) / Z
        w = exp_ov / Z
        
        grad_h = np.zeros(N)
        
        # Contribution from first term: sum_{k: j < N-k} w_k (1 - h_{j+k})
        # j < N-k => k < N-j
        # For a fixed j, k ranges from 0 to N-j-1.
        # Let's loop over j? N=100, N^2=10000 ops per iter. 5000 iters = 50M ops. Fast enough.
        
        for j in range(N):
            # Term 1: k < N-j
            if N - j > 0:
                # k in 0..N-j-1
                # Sum w[k] * (1 - h[j+k])
                # h[j+k] for k=0..N-j-1 is h[j:N]
                term1 = np.sum(w[:N-j] * (1.0 - h[j:N]))
            else:
                term1 = 0.0
            
            # Term 2: k <= j (and k < N, which is true)
            # Sum w[k] * (-h[j-k])
            # k in 0..j
            # h[j-k] for k=0..j is h[j::-1][:j+1] i.e. h[0..j] reversed
            term2 = -np.sum(w[:j+1] * h[j::-1][:j+1])
            
            grad_h[j] = term1 + term2
        
        # Gradient step
        h_new = h - lr * grad_h
        
        # Project
        h_new = _project(h_new, N / 2.0)
        
        # Check convergence or improvement
        new_ov = _overlaps(h_new)
        new_max = np.max(new_ov)
        
        if new_max < best_val:
            best_val = new_max
            best_h = h_new.copy()
            
        h = h_new
        
        # Adaptive learning rate?
        # If stuck, reduce lr?
        # Simple decay
        lr = 0.1 / (1.0 + it * 0.001)
        
    return best_h
