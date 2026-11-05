import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 148.0

def _evaluate(h):
    """
    Compute the task value.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    # np.max takes the maximum signed entry.
    # The value is abs(2 * N * max_conv / s^2)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """
    Objective: max(conv(h, h)) / sum(h)^2
    We minimize this. Note that the evaluator takes abs(), so we can assume
    we are in a region where the max is positive (which is usually the case 
    for non-constant signals due to the main lobe at lag 0, unless sum is 
    large and negative/positive such that other peaks dominate, but typically 
    lag 0 is the peak).
    
    Actually, let's look at the structure.
    conv(h, h)[k] = sum h[i] h[i+k]
    max is usually at k=0, which is sum(h^2).
    If we minimize sum(h^2) / sum(h)^2, we are minimizing the "peakiness" or 
    maximizing the uniformity.
    
    However, the evaluator uses np.max(conv(h, h)).
    For a signal with large sum, the peak at lag 0 is sum(h^2).
    If the signal is constant h[i]=c, sum(h)=Nc, sum(h^2)=Nc^2.
    value = 2*N*Nc^2 / (Nc)^2 = 2.
    
    If we can make the autoconvolution flat, i.e., conv(h, h) = C for all lags,
    then max = C.
    sum(conv(h, h)) = sum(h)^2.
    Length of full conv is 2N-1.
    If flat, C = sum(h)^2 / (2N-1).
    value = 2*N * (sum(h)^2 / (2N-1)) / sum(h)^2 = 2N / (2N-1) ~ 1.0025.
    
    So the goal is to make conv(h, h) as flat as possible.
    
    Let f(h) = max_k conv(h, h)_k / (sum(h))^2.
    This is differentiable almost everywhere.
    Let M(h) = max_k conv_k.
    Let S(h) = sum(h).
    f = M / S^2.
    df = (S^2 dM - M * 2S dS) / S^4 = (dM/S - 2M dS/S^2) / S.
    
    dM/dh:
    M is the max of a set of functions. The subgradient is a convex combination 
    of the gradients of the active terms. For L-BFGS-B, we can approximate 
    the gradient by taking the gradient of the maximum term, or a smoothed max.
    Using the "soft max" (log-sum-exp) is differentiable everywhere.
    Let L = log-sum-exp(conv(h, h) / tau) * tau.
    As tau -> 0, L -> max.
    Gradient of L w.r.t h:
    Let z_k = conv_k.
    p_k = exp(z_k/tau) / sum(exp(z_j/tau)).
    dL/dz_k = p_k.
    dz_k/dh_i = h_{i-k} + h_{i+k} (with boundary checks).
    
    Alternatively, just use the gradient of the actual max term.
    Since we are optimizing, the max term is likely unique or we can pick one.
    If we pick the index k* where conv is max, 
    d(conv_k*)/dh_i = h_{i-k*} + h_{i+k*} (where valid).
    
    Let's implement the soft-max gradient for robustness.
    """
    h = np.asarray(h, dtype=np.float64)
    S = np.sum(h)
    if abs(S) < 1e-10:
        return 1e10, np.zeros(N)
    
    conv = np.convolve(h, h)
    # Soft max parameter
    tau = 0.1 # Small enough to approximate max, large enough for stability
    
    # Log-sum-exp trick for stability
    max_conv = np.max(conv)
    log_sum_exp = max_conv + tau * np.log(np.sum(np.exp((conv - max_conv) / tau)))
    M_soft = log_sum_exp
    
    # We want to minimize M_soft / S^2
    # Let's compute gradient of M_soft / S^2
    
    # Probabilities for soft max
    p = np.exp((conv - max_conv) / tau)
    p /= np.sum(p)
    
    # Gradient of M_soft w.r.t conv is p
    # Gradient of conv_k w.r.t h_i
    # conv_k = sum_j h_j h_{j+k}
    # d(conv_k)/dh_i = h_{i-k} + h_{i+k}
    
    # We need dM_soft/dh = sum_k p_k * d(conv_k)/dh
    # This is a weighted sum of shifted h's.
    # d(conv_k)/dh_i is non-zero when i-k in [0, N-1] or i+k in [0, N-1].
    
    # Efficient computation:
    # grad_M = sum_k p_k * (h shifted by -k + h shifted by k)
    # This is a convolution-like operation.
    # Let's construct a kernel G where G[k] = p_k.
    # grad_M = convolve(h, G, mode='same')? No, the shifts are specific.
    
    # d(conv_k)/dh_i = h[i-k] + h[i+k]
    # grad_M[i] = sum_k p_k (h[i-k] + h[i+k])
    # Let v = h.
    # Term 1: sum_k p_k h[i-k]. This is convolution of h and p (reversed p).
    # Term 2: sum_k p_k h[i+k]. This is correlation of h and p.
    
    # p is defined on indices 0..2N-2 of conv.
    # Let's map p to indices relative to center?
    # conv index k corresponds to lag k.
    # h indices 0..N-1.
    
    # Let's just compute it directly for N=400, it's fast enough.
    # grad_M = zeros(N)
    # for k in range(2*N-1):
    #    if p[k] > 1e-8:
    #       # h[i-k] term: i-k in [0, N-1] => i in [k, k+N-1]
    #       # valid i: max(0, k) to min(N-1, k+N-1)
    #       # h[i] contributes to grad_M[i+k] ? No.
    #       # d(conv_k)/dh_i = h[i-k] + h[i+k]
    #       # So for a fixed k, the vector d(conv_k)/dh is h shifted.
    #       # Specifically, (d(conv_k)/dh)_i = h[i-k] (if valid) + h[i+k] (if valid).
    #       # This vector is h[-k : N-k] padded + h[k : N+k] padded?
    #       # Let's use np.roll or slicing.
    #       pass
    
    # Vectorized approach:
    # grad_M[i] = sum_{k} p[k] * (h[i-k] + h[i+k])
    # Let's define an array P_ext of size 2N-1 containing p.
    # We can compute this as a convolution of h with a symmetric kernel?
    # Note p[k] is not necessarily symmetric.
    
    # Let's compute the two parts separately.
    # Part 1: sum_k p[k] h[i-k]
    # This is the cross-correlation of h and p?
    # Let's use scipy.signal.fftconvolve for speed if needed, but N=400 is small.
    # Direct loop over k is O(N^2) = 160k ops. Very fast.
    
    grad_M = np.zeros(N)
    # p is length 2N-1
    for k in range(2 * N - 1):
        if p[k] > 1e-9:
            # Term h[i-k]: i-k in [0, N-1] => i in [k, k+N-1]
            # Intersection with [0, N-1] is [max(0,k), min(N-1, k+N-1)]
            # Since k >= 0, start = k. End = min(N-1, k+N-1) = N-1 if k < N, else ...
            # Actually i is in 0..N-1.
            # If k >= N, i-k >= 0 => i >= k >= N, impossible. So h[i-k] is 0 for k >= N.
            # So only k < N contributes to h[i-k].
            if k < N:
                # i ranges from k to N-1
                # h[i-k] ranges from h[0] to h[N-1-k]
                # grad_M[i] += p[k] * h[i-k]
                # grad_M[k:N] += p[k] * h[0:N-k]
                grad_M[k:N] += p[k] * h[0:N-k]
            
            # Term h[i+k]: i+k in [0, N-1] => i in [-k, N-1-k]
            # Intersection with [0, N-1] is [0, min(N-1, N-1-k)] = [0, N-1-k]
            # So only k < N contributes?
            # If k >= N, N-1-k < 0, so no i.
            # So only k < N contributes to h[i+k] as well?
            # Wait. conv index k goes up to 2N-2.
            # If k is large, say 2N-2.
            # h[i+k] with i=0 is h[2N-2], out of bounds.
            # So yes, for h[i+k] to be valid, i+k <= N-1 => k <= N-1-i <= N-1.
            # So only k in [0, N-1] contributes to the gradient?
            
            # Let's re-verify.
            # conv_k = sum_{j} h_j h_{j+k}
            # j ranges such that 0<=j<N and 0<=j+k<N.
            # So j in [max(0, -k), min(N-1, N-1-k)].
            # If k > 0: j in [0, N-1-k].
            # If k < 0 (not in our conv index range 0..2N-2, but mathematically):
            # Our conv indices are 0..2N-2.
            # k=0: j in [0, N-1].
            # k=1: j in [0, N-2]. h_j h_{j+1}.
            # ...
            # k=N-1: j in [0, 0]. h_0 h_{N-1}.
            # k=N: j in [0, -1]. Empty.
            
            # So conv_k is 0 for k >= N?
            # np.convolve(h, h) length is 2N-1.
            # h=[1,2], conv=[1,4,4]. N=2. 2N-1=3.
            # k=0: 1*1=1.
            # k=1: 1*2+2*1=4.
            # k=2: 2*2=4.
            # So conv_k is non-zero for k up to 2N-2.
            
            # My index derivation:
            # conv_k = sum_j h_j h_{j+k} ?
            # Standard conv definition: (a*b)[n] = sum_m a_m b_{n-m}.
            # np.convolve(h, h)[n] = sum_m h_m h_{n-m}.
            # Let k = n.
            # h_m is non-zero for m in [0, N-1].
            # h_{k-m} is non-zero for k-m in [0, N-1] => m in [k-(N-1), k].
            # Intersection of [0, N-1] and [k-N+1, k].
            
            # If k < 0: empty (n>=0).
            # If 0 <= k < N:
            #   m in [0, N-1] and m in [k-N+1, k].
            #   Since k-N+1 <= 0 (as k<N), lower bound is 0.
            #   Upper bound is min(N-1, k) = k.
            #   So m in [0, k].
            #   conv_k = sum_{m=0}^k h_m h_{k-m}.
            
            # If N <= k <= 2N-2:
            #   m in [0, N-1] and m in [k-N+1, k].
            #   Since k>=N, k-N+1 >= 1.
            #   Lower bound is k-N+1.
            #   Upper bound is min(N-1, k) = N-1.
            #   So m in [k-N+1, N-1].
            #   conv_k = sum_{m=k-N+1}^{N-1} h_m h_{k-m}.
            
            # Gradient d(conv_k)/dh_i:
            # Case 1: 0 <= k < N.
            # conv_k = sum_{m=0}^k h_m h_{k-m}.
            # d/dh_i:
            # If i in [0, k]:
            #   Term m=i: h_i h_{k-i} -> h_{k-i}
            #   Term m=k-i (if k-i != i): h_{k-i} h_i -> h_i
            #   So if i != k-i (i != k/2), derivative is h_{k-i} + h_i.
            #   If i == k/2 (k even), derivative is 2 h_i.
            
            # Case 2: N <= k <= 2N-2.
            # Let k' = 2N-2-k. Note k' < N.
            # conv_k = conv_{k'} (symmetry of autoconv).
            # So gradient is same as for k'.
            
            # So we can just use symmetry.
            # p[k] contributes to gradient.
            # Let's compute gradient contribution for each k.
            
            # Optimization: Use the symmetry conv[k] = conv[2N-2-k].
            # p[k] and p[2N-2-k] might be different if the max is not symmetric?
            # But conv is symmetric. The soft-max p depends on conv values.
            # If conv is symmetric, p is symmetric.
            # So we can just sum over k < N and double?
            # No, p is computed on the full array.
            
            # Let's just do the direct calculation for k in 0..N-1.
            # For k in 0..N-1:
            #   d(conv_k)/dh_i is non-zero for i in 0..k.
            #   Specifically, vector v_k of length N:
            #   v_k[i] = h[k-i] + h[i] for i < k/2
            #   v_k[i] = 2*h[i] for i == k/2 (if k even)
            #   v_k[i] = h[i] + h[k-i] for i > k/2
            #   Actually, it's just h[k-i] + h[i] for all i in 0..k, where h[j]=0 if j out of [0,N-1].
            #   Since i <= k < N, k-i >= 0 and < N. So both terms are valid.
            
            # So for k in 0..N-1:
            #   grad_M[0:k+1] += p[k] * (h[k::-1] + h[0:k+1])
            #   Wait, h[k::-1] is h[k], h[k-1]...h[0].
            #   h[0:k+1] is h[0]...h[k].
            #   Sum is h[k]+h[0], h[k-1]+h[1]...
            
            # What about k >= N?
            # conv_k = conv_{2N-2-k}.
            # Let k_sym = 2N-2-k.
            # p[k] * d(conv_k)/dh = p[k] * d(conv_{k_sym})/dh.
            # So we can add p[k] to an effective weight for k_sym.
            
            # Let's create an array w of length N (for k=0..N-1).
            # w[k] = p[k] + p[2N-2-k]
            # Then grad_M = sum_{k=0}^{N-1} w[k] * (h[k::-1] + h[0:k+1])
            
            # This is O(N^2) to compute w and the sum.
            # N=400, 160k ops. Fine.
            
            pass # We will implement this logic outside the loop for efficiency

    # Re-implementing gradient computation efficiently
    # w[k] = p[k] + p[2N-2-k] for k in 0..N-1
    w = p[:N] + p[N-1::-1] # p[2N-2-k] for k=0 is p[2N-2], which is p[-1]. p[N-1::-1] is p[N-1], p[N-2]...p[0].
    # Wait.
    # k=0 -> sym = 2N-2. p[2N-2].
    # k=1 -> sym = 2N-3.
    # ...
    # k=N-1 -> sym = N-1. p[N-1].
    # So w[k] = p[k] + p[2N-2-k].
    # p[2N-2-k] for k=0..N-1 corresponds to indices 2N-2, 2N-3, ..., N-1.
    # This is p[N-1::-1] ?
    # p[N-1::-1] is p[N-1], p[N-2], ..., p[0].
    # We need p[2N-2], ..., p[N-1].
    # This is p[-1:N-2:-1] or p[2*N-2:N-2:-1].
    
    p_sym = p[2*N-2 : N-2 : -1]
    w = p[:N] + p_sym
    
    # Now compute grad_M = sum_{k=0}^{N-1} w[k] * (h[k::-1] + h[0:k+1])
    # This is a weighted sum of "triangular" vectors.
    # We can compute this using convolution or cumsum tricks, but N=400 is small.
    # Let's use a loop, it's clear and fast enough.
    grad_M = np.zeros(N)
    for k in range(N):
        if w[k] != 0:
            # Vector v of length k+1
            # v[i] = h[k-i] + h[i]
            # h[k::-1] is h[k], h[k-1]...h[0]
            # h[0:k+1] is h[0]...h[k]
            # Sum:
            # grad_M[0:k+1] += w[k] * (h[k::-1] + h[0:k+1])
            grad_M[0:k+1] += w[k] * (h[k::-1] + h[0:k+1])
    
    # Gradient of S^2
    # d(S^2)/dh = 2S * 1 (vector of ones)
    
    # Objective f = M_soft / S^2
    # df = (S^2 grad_M - M_soft * 2S * 1) / S^4
    #    = grad_M / S - 2 M_soft / S^2 * 1
    
    grad_f = grad_M / S - (2.0 * M_soft / (S * S)) * np.ones(N)
    
    val = M_soft / (S * S)
    # The evaluator uses abs(). If val is negative, we should minimize -val?
    # M_soft is max of conv. Conv of real signal can be negative?
    # Conv at lag 0 is sum(h^2) >= 0. So max >= 0.
    # So M_soft >= 0.
    # S^2 > 0.
    # So val >= 0.
    # We minimize val.
    
    return val, grad_f

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(42)
    
    # Initial candidates
    # 1. Constant (seed)
    # 2. Random
    # 3. Chirp (from previous knowledge, good starting point)
    
    candidates = [np.ones(N)]
    
    # Random
    h_rand = rng.normal(0, 1, N)
    h_rand /= np.sum(h_rand) # Normalize to sum 1 to avoid scaling issues?
    # Actually, the objective is scale invariant?
    # f(c*h) = max(conv(c*h,c*h)) / (sum(c*h))^2 = c^2 max(conv) / c^2 sum^2 = f(h).
    # So scale doesn't matter. We can fix sum(h)=1 to reduce dimensionality?
    # L-BFGS-B handles it.
    
    candidates.append(h_rand)
    
    # Chirp: h[n] = cos(2*pi*f*n^2/N + phase)
    # Try a few frequencies
    for f in [1, 2, 5, 10]:
        n = np.arange(N)
        h_chirp = np.cos(2 * np.pi * f * n * n / N)
        candidates.append(h_chirp)
        
    # Sum of two chirps
    n = np.arange(N)
    h_chirp1 = np.cos(2 * np.pi * 1 * n * n / N)
    h_chirp2 = np.cos(2 * np.pi * 3 * n * n / N + np.pi/4)
    candidates.append(h_chirp1 + h_chirp2)
    
    # Optimization loop
    for h_init in candidates:
        if time.time() > deadline - 20:
            break
            
        h_init = np.asarray(h_init, dtype=np.float64)
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
        
        # Normalize to sum=1 for stability in optimization?
        # It's scale invariant, but L-BFGS might struggle if values are huge/tiny.
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(
                _objective_and_grad,
                h_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            h_opt = res.x
            val_opt = _evaluate(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
        # Check time
        if time.time() > deadline - 10:
            break

    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# Quick test to ensure it runs
if __name__ == "__main__":
    h = construct()
    print(_evaluate(h))
