import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute overlaps for all linear shifts k=0..N-1.
    O_k = SCALE * sum_{j=k}^{N-1} h[j] * (1 - h[j-k])
    """
    O = np.zeros(N)
    # Vectorized calculation
    # S1[k] = sum_{j=k}^{N-1} h[j]
    # S2[k] = sum_{j=k}^{N-1} h[j] * h[j-k]
    
    # S1 can be computed via cumulative sum from the end
    csum = np.cumsum(h[::-1])[::-1]
    S1 = csum
    
    # S2[k] is dot product of h[k:] and h[:N-k]
    # We can use a loop for N=100, it's fast enough (100 * 100 ops = 10k)
    # Or use FFT for circular, but this is linear.
    # Loop is safe and simple.
    for k in range(N):
        O[k] = SCALE * (S1[k] - np.dot(h[k:], h[:N - k]))
        
    return O

def _grad_soft_max(h, tau):
    """
    Compute gradient of the soft-max objective:
    S = tau * log(sum(exp(O_k / tau)))
    grad S = sum_k w_k * grad O_k
    where w_k = exp(O_k / tau) / sum(exp(O_j / tau))
    """
    O = _compute_O(h)
    # Stability for exp
    O_max = np.max(O)
    z = (O - O_max) / tau
    ez = np.exp(z)
    w = ez / np.sum(ez)
    
    # Compute gradient of each O_k and accumulate weighted sum
    # grad O_k[i] = SCALE * ( 1_{k<=i<N} - 1_{k<=i<N} h[i-k] - 1_{0<=i<N-k} h[i+k] )
    
    grad = np.zeros(N)
    
    # We can vectorize the accumulation over k.
    # grad[i] = SCALE * sum_k w_k * ( 1_{k<=i} - 1_{k<=i} h[i-k] - 1_{0<=i<N-k} h[i+k] )
    
    # Term 1: sum_k w_k * 1_{k<=i} = sum_{k=0}^i w_k
    # Let W_csum = cumsum(w)
    W_csum = np.cumsum(w)
    # For a given i, sum_{k=0}^i w_k is W_csum[i]
    term1 = SCALE * W_csum
    
    # Term 2: sum_k w_k * 1_{k<=i} h[i-k]
    # This is a convolution of w and h?
    # Let j = i-k. k = i-j.
    # Sum over k such that k<=i and k<N. Also j=i-k >= 0.
    # The condition k<=i is always true for j>=0 if k=i-j.
    # The condition k<N is i-j < N => j > i-N.
    # The condition k>=0 is i-j >= 0 => j <= i.
    # Also from the definition of O_k, the sum is over j=k..N-1.
    # The term is h[i-k]. Let's stick to the loop for clarity and correctness, N=100 is small.
    
    # Loop over k for gradient accumulation
    # This is O(N^2) per gradient step. 100^2 = 10,000 ops.
    # With 1000 steps, 10^7 ops. Very fast in Python/NumPy if we can vectorize inside.
    
    # Let's try to vectorize the inner loop.
    # grad[i] = SCALE * ( W_csum[i] - sum_{k=0}^i w_k h[i-k] - sum_{k=0}^{N-1-i} w_k h[i+k] )
    
    # Second term: sum_{k=0}^i w_k h[i-k]
    # This is the linear convolution of w and h, evaluated at i, but only up to i.
    # Since w is defined for 0..N-1, and we sum k=0..i, this is (w * h)[i] if we treat them as sequences starting at 0.
    # Actually, np.convolve(w, h, mode='full') gives sums for all i.
    # conv[i] = sum_{k} w[k] h[i-k].
    # We need sum_{k=0}^i w[k] h[i-k].
    # Since h[j] is 0 for j<0 or j>=N, and w[k] is 0 for k<0 or k>=N.
    # The term h[i-k] is non-zero only if 0 <= i-k < N => i-N < k <= i.
    # The sum in the gradient formula is over k such that the term exists.
    # The formula derived: 1_{k<=i<N} h[i-k].
    # So k must be <= i.
    # Also i-k must be in [0, N-1].
    # So k >= i-(N-1).
    # So sum is over k in [max(0, i-N+1), i].
    
    # np.convolve(w, h, 'full')[i] sums over all k where w[k] and h[i-k] are valid.
    # This matches the range exactly because w and h are zero-padded effectively.
    # So term2 = SCALE * np.convolve(w, h, mode='full')[i] ?
    # Let's check indices.
    # conv_full[i] = sum_{k=0}^{N-1} w[k] h[i-k].
    # If i-k < 0, h[i-k] is 0 (conceptually).
    # If i-k >= N, h[i-k] is 0.
    # So yes, it matches.
    
    conv_wh = np.convolve(w, h, mode='full')
    
    # Third term: sum_{k=0}^{N-1-i} w_k h[i+k]
    # Let j = i+k. k = j-i.
    # Sum over k such that 0 <= k <= N-1-i.
    # So j ranges from i to N-1.
    # Term is w[j-i] h[j].
    # This is a cross-correlation or convolution with reversed w?
    # sum_j h[j] w[j-i].
    # This is (h * w_reversed)[?].
    # Let's just use the property: sum_{k} w[k] h[i+k] = sum_{j} w[j-i] h[j].
    # This is the cross-correlation of h and w at lag i?
    # Correlation R_xy[i] = sum_j x[j] y[j-i].
    # Here we have sum_j h[j] w[j-i].
    # This is np.correlate(h, w, mode='full')?
    # np.correlate(a, v, 'full') computes sum_k a[k] v[k-i] (or similar).
    # Let's check numpy docs.
    # correlate(a, v) = sum_k a[k] v[k-i].
    # We want sum_k w[k] h[i+k]. Let m = i+k. k = m-i.
    # Sum_m w[m-i] h[m].
    # This is sum_m h[m] w[m-i].
    # This is correlate(h, w, 'full') at index corresponding to lag i?
    # np.correlate(h, w, 'full')[N-1 + i] ?
    # Let's verify.
    # correlate(a, v, 'full')[k] = sum_j a[j] v[j - (k - (len(v)-1))] ? No, it's messy.
    
    # Given N=100, a simple loop for the third term or even the whole gradient is acceptable if optimized.
    # But let's try to be efficient.
    
    # Let's compute the third term using a loop over i, which is O(N^2) total if we do it naively, 
    # but we can precompute.
    
    # Actually, let's just use the loop for the gradient accumulation in the FISTA step.
    # It's only 100 iterations of length 100.
    
    # Re-implementing gradient with explicit loops for clarity and correctness first, 
    # then optimize if needed.
    
    grad = np.zeros(N)
    for k in range(N):
        if w[k] < 1e-9:
            continue
        # grad contribution from O_k
        # indices i where k <= i < N
        idx1 = np.arange(k, N)
        # h[i-k] for i in idx1 -> h[0 : N-k]
        term1_k = SCALE * np.ones(N)
        term1_k[:k] = 0.0
        
        term2_k = np.zeros(N)
        term2_k[k:] = h[:N-k]
        
        term3_k = np.zeros(N)
        # indices i where 0 <= i < N-k
        # h[i+k] for i in 0..N-1-k -> h[k : N]
        term3_k[:N-k] = h[k:]
        
        grad += w[k] * (term1_k - term2_k - term3_k)
        
    return O_max + tau * np.log(np.sum(ez)) + O_max, grad # Objective value and gradient
    # Note: S = tau * log(sum exp((O-Omax)/tau)) + Omax

def _project(z):
    """
    Project z onto {h : 0 <= h_i <= 1, sum h_i = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    
    # Bisection for threshold t
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _fista_anneal(h_init, deadline):
    h = _project(h_init.copy())
    h_prev = h.copy()
    t_mom = 1.0
    
    # Annealing schedule for tau
    # Start large, end small
    taus = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    L = 5.0 # Initial Lipschitz constant estimate
    
    for tau in taus:
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Run FISTA for a fixed number of steps for this tau
        # More steps for smaller tau? Or just fixed?
        # Let's allocate time budget.
        # 10 taus. 
        steps_for_tau = 200
        
        for i in range(steps_for_tau):
            if time.time() > deadline - 2.0:
                break
            
            obj, grad = _grad_soft_max(h, tau)
            
            # Backtracking line search
            step = 1.0 / L
            y = h - step * grad
            y_proj = _project(y)
            
            # Check decrease in soft-max objective
            obj_new, _ = _grad_soft_max(y_proj, tau)
            
            if obj_new < obj:
                h = y_proj
                L = L / 1.5
            else:
                L = L * 1.5
                continue
            
            # FISTA momentum
            t_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
            h_prev = h.copy()
            h = _project(h + ((t_mom - 1) / t_new) * (h - h_prev))
            t_mom = t_new
            
    return h

def construct():
    start = time.time()
    deadline = start + 145.0 # Safety margin for 150s limit
    
    best_h = np.full(N, 0.5)
    best_val = float(_compute_O(best_h).max())
    
    # Initializations
    inits = [best_h.copy()]
    
    # Structured inits
    idx = np.arange(N)
    for f in (1, 2, 3, 5, 8, 13):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.4 * np.sin(2.0 * np.pi * f * idx / N))
        
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.random(N))
        
    # Optimize each init
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        
        try:
            h_opt = _fista_anneal(init, deadline)
            val = float(_compute_O(h_opt).max())
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            continue
            
    return best_h
