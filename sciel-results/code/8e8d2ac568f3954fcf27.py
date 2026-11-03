"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time

# EVOLVE-BLOCK-START
def _autoconvolution_max_abs(h):
    """Compute the maximum absolute value of the linear autoconvolution of h."""
    # Linear autoconvolution
    ac = np.convolve(h, h, mode='full')
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective: max_abs_autoconv / (integral^2). Lower is better."""
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        return 1e10
    return _autoconvolution_max_abs(h) / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Time-reversal symmetric representation: h[i] = h[N-1-i]
    # We only optimize the first half (or first half + 1 if N is odd)
    half_len = N // 2 + (N % 2)
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Initialize with symmetric random vector
    np.random.seed(42)
    h_half = np.random.randn(half_len)
    # Normalize to have unit sum for stability
    h_half = h_half / np.sum(h_half)
    
    def expand(h_half):
        """Expand symmetric half-vector to full vector."""
        if N % 2 == 0:
            return np.concatenate([h_half, h_half[::-1]])
        else:
            # Middle element is shared
            return np.concatenate([h_half, h_half[-2::-1]])
    
    h = expand(h_half)
    current_score = _score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
    
    # Projected gradient descent with soft-max surrogate
    # Objective: minimize max(|ac|) / (sum(h))^2
    # Use soft-max: log(sum(exp(beta * |ac|))) / beta as surrogate for max(|ac|)
    
    beta = 10.0  # Sharpness of soft-max
    lr = 0.01
    momentum = 0.9
    v = np.zeros(half_len)
    
    # Pre-compute convolution indices for efficiency
    # For symmetric h, autoconvolution has special structure
    
    for step in range(5000):
        if time.time() > deadline - 5.0:
            break
            
        # Expand to full vector
        h = expand(h_half)
        
        # Compute autoconvolution
        ac = np.convolve(h, h, mode='full')
        ac_abs = np.abs(ac)
        
        # Soft-max surrogate for max
        # Shift for numerical stability
        max_ac = np.max(ac_abs)
        exp_ac = np.exp(beta * (ac_abs - max_ac))
        log_sum_exp = max_ac + np.log(np.sum(exp_ac)) / beta
        
        # Integral
        integral = np.sum(h)
        if abs(integral) < 1e-12:
            break
        
        # Objective value
        obj = log_sum_exp / (integral * integral)
        
        # Gradient with respect to h_half
        # d/dh_half [log_sum_exp / integral^2]
        # = (1/integral^2) * d(log_sum_exp)/dh_half - 2*log_sum_exp/integral^3 * d(integral)/dh_half
        
        # d(integral)/dh_half: each h_half[i] contributes to 2 positions (or 1 for middle if odd)
        d_int = np.ones(half_len) * 2.0
        if N % 2 == 1:
            d_int[-1] = 1.0  # Middle element appears only once
        
        # d(log_sum_exp)/dh_half
        # log_sum_exp = max_ac + log(sum(exp(beta*(ac_abs - max_ac)))) / beta
        # d(ac_abs)/dh_half: need gradient of |ac| w.r.t. h_half
        # ac = convolve(h, h), so d(ac[k])/dh[i] = h[k-i] + h[k-i] (by symmetry of conv)
        # Actually d(conv(h,h)[k])/dh[i] = h[k-i] + h[k-i] if k-i is valid, accounting for symmetry
        
        # For efficiency, compute gradient numerically or analytically
        # Let's use a simplified approach: compute gradient of soft-max over ac
        
        # weights for soft-max
        weights = exp_ac / np.sum(exp_ac)  # sum to 1
        
        # d(log_sum_exp)/dac_abs = beta * weights
        # d(ac_abs)/dac = sign(ac)
        # d(ac[k])/dh[i] = sum over j where h[j]*h[k-j] contributes
        # For conv(h,h)[k] = sum_j h[j]*h[k-j], d/dh[i] = h[k-i] + h[k-i] (if i != k-i)
        # = 2*h[k-i] if i != k-i, h[k] if i == k-i
        
        # This is complex; let's use finite differences for the gradient of the surrogate
        # But that's slow. Let's use a simpler heuristic: 
        # Reduce peaks by pushing down h where ac is large
        
        # Alternative: use the fact that for symmetric sequences, we can work in frequency domain
        # |autoconv| in time = |FFT(h)|^2 in frequency (for circular conv)
        # For linear conv, it's more complex
        
        # Let's try a simpler approach: iterative peak reduction
        # Find where ac_abs is largest, reduce h at corresponding positions
        
        # For now, use a simple gradient step based on the correlation structure
        # The gradient of max|ac| w.r.t h[i] is proportional to h convolved with sign(ac) at peak
        
        # Simplified: move h_half in direction that reduces the soft-max objective
        # Approximate gradient: for each peak location k in ac, the contribution to h[i] 
        # is related to h[k-i]
        
        # Let's use a direct approach: compute gradient via automatic-like method
        # d(log_sum_exp)/dh[i] = sum_k beta * weights[k] * sign(ac[k]) * d(ac[k])/dh[i]
        # d(ac[k])/dh[i] = sum_j delta(i,j)h[k-j] + delta(i,k-j)h[j] = h[k-i] + h[k-i] (with bounds)
        
        # For symmetric h, this simplifies
        # Let's just use a simple update: reduce h_half where it correlates with peak ac regions
        
        # Compute correlation between h_half and the "peak pattern"
        # For each peak at position k in ac (length 2N-1), it involves h[j]*h[k-j]
        # The most direct influence is on h_half indices that pair up to k
        
        # Simplified gradient: 
        # For symmetric h, ac[k] = sum_j h[j]h[k-j]
        # d(ac[k])/dh_half[i] = 2*h_half[k-i] (if valid) approximately
        
        # Let's compute the gradient numerically for a few key peaks
        # But to stay within time, let's use a heuristic update
        
        # Heuristic: identify the top few peak locations in ac
        top_k = np.argsort(ac_abs)[-10:]
        
        # For each peak, determine which h_half indices to reduce
        grad = np.zeros(half_len)
        for k in top_k:
            w = weights[k] * beta * np.sign(ac[k])
            # k ranges from 0 to 2N-2
            # h[j]*h[k-j] contributes, j from max(0,k-N+1) to min(N-1,k)
            # For symmetric h, h[j]=h[N-1-j]
            # The indices in h_half that contribute: j and N-1-j
            j_start = max(0, k - N + 1)
            j_end = min(N - 1, k)
            for j in range(j_start, j_end + 1):
                # j maps to h_half index
                if j < half_len:
                    hidx = j
                else:
                    hidx = N - 1 - j
                    if hidx >= half_len:
                        continue
                # Contribution: w * h[j'] where j' = k-j
                jp = k - j
                if jp < 0 or jp >= N:
                    continue
                # h[jp] in terms of h_half
                if jp < half_len:
                    hval = h_half[jp]
                else:
                    hval = h_half[N - 1 - jp]
                    if N - 1 - jp >= half_len:
                        continue
                # d(ac[k])/dh_half[hidx] includes h[jp] (and possibly another term)
                # For simplicity, add w * hval
                grad[hidx] += w * hval
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Momentum update
        v = momentum * v - lr * grad
        h_half_new = h_half + v
        
        # Project: maintain non-zero sum (scale to unit sum)
        s = np.sum(h_half_new)
        if abs(s) > 1e-12:
            h_half_new = h_half_new / s
        
        h_half = h_half_new
        
        # Check score
        h_full = expand(h_half)
        sc = _score(h_full)
        if sc < best_score:
            best_score = sc
            best_h = h_full.copy()
        
        # Adaptive learning rate
        if step % 100 == 0 and step > 0:
            lr *= 0.95
    
    return best_h
# EVOLVE-BLOCK-END
