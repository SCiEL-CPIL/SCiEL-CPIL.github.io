"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 600
STEP = 1.0 / N  # step size for intervals on [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of 600 nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of the autoconvolution (upper bound on C1).
    
    Strategy:
    1. Start with flat function (theoretical lower bound by Cauchy-Schwarz).
    2. Use efficient FFT-based gradient computation for projected gradient descent.
    3. Use log-sum-exp soft-max surrogate for the max constraint.
    4. If optimization fails or doesn't improve, return flat function.
    """
    t0 = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Flat function: achieves Cauchy-Schwarz lower bound
    flat = np.ones(N)
    
    # Compute the objective for flat function to establish baseline
    def autoconv_max_normalized(f):
        """Compute max of autoconvolution normalized by (sum f)^2 * N."""
        f = np.asarray(f, dtype=np.float64)
        if np.any(f < -1e-12):
            return np.inf
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution via FFT
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        max_ac = np.max(ac)
        return max_ac / (s * s) * N
    
    baseline_val = autoconv_max_normalized(flat)
    best_f = flat.copy()
    best_val = baseline_val
    
    # Efficient gradient computation using FFT
    # ac[k] = sum_{j=0}^{N-1-k} f[j] * f[j+k]  (linear autocorrelation)
    # d(ac[k])/df[i] = f[i+k] (if i+k < N) + f[i-k] (if i-k >= 0)
    # This can be computed as: grad_f = weights_conv * f + flip(weights_conv) * f
    # where weights_conv is the softmax weights convolved appropriately
    
    def compute_ac(f):
        """Compute linear autocorrelation via FFT."""
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        return ac
    
    def compute_grad_ac(f, weights):
        """
        Compute gradient of sum_k weights[k] * ac[k] w.r.t. f.
        ac[k] = sum_j f[j]*f[j+k]
        d(sum_k w_k * ac_k)/df[i] = sum_k w_k * (f[i+k] + f[i-k]) [with bounds]
        This equals: (w * f) + (flip(w) * f) where * denotes linear convolution
        """
        # For each i: sum_k w_k * (f[i+k] if i+k<N else 0 + f[i-k] if i-k>=0 else 0)
        # = sum_k w_k * f[i+k] + sum_k w_k * f[i-k]
        # First term: (w reversed) * f at position i... let me think carefully
        # sum_k w_k * f[i+k] for k where i+k < N
        # Let j = i+k, so k = j-i: sum_{j=i}^{N-1} w_{j-i} * f[j]
        # This is a cross-correlation of w and f
        # Second term: sum_k w_k * f[i-k] for k where i-k >= 0
        # Let j = i-k, so k = i-j: sum_{j=0}^{i} w_{i-j} * f[j]
        # This is also related to correlation
        
        # Use FFT for efficiency
        nfft = 2 * N
        W = np.fft.rfft(weights, n=nfft)
        F = np.fft.rfft(f, n=nfft)
        
        # First term: sum_k w_k * f[i+k] 
        # = sum_j f[j] * w[j-i] for j >= i
        # This is like (f * reverse(w))[i] but need to be careful
        # Actually: sum_k w_k * f[i+k] = (f cross-correlate w)[i] where we define
        # (f ⋆ w)[i] = sum_k f[i+k] * w[k]
        # Using FFT: this is irfft(conj(rfft(w)) * rfft(f)) but with proper alignment
        
        # Let me use a different approach: 
        # grad[i] = sum_k w_k * (f[i+k] + f[i-k]) with boundary handling
        # = sum_k w_k * f[i+k] + sum_k w_k * f[i-k]
        
        # For the first sum: let's define g1[i] = sum_{k=0}^{N-1-i} w_k * f[i+k]
        # This is the cross-correlation of f and w evaluated at i
        # Using FFT: g1 = ifft(conj(fft(w)) * fft(f))[:N] but need linear correlation
        
        # Simpler: use np.correlate or manual FFT
        # g1[i] = sum_k w[k] * f[i+k] for valid k
        # = (f * w_flipped)[i + N - 1] if we use linear convolution
        # where w_flipped = reverse(w)
        
        w_flipped = weights[::-1].copy()
        # Linear convolution of f and w_flipped
        conv_fw = np.fft.irfft(np.fft.rfft(f, n=nfft) * np.fft.rfft(w_flipped, n=nfft), n=nfft)
        # conv_fw[j] = sum_i f[i] * w_flipped[j-i] = sum_i f[i] * w[N-1-(j-i)] = sum_i f[i] * w[N-1-j+i]
        # We want g1[i] = sum_k w[k] * f[i+k] = sum_k f[i+k] * w[k]
        # Let j = i+k, k = j-i: g1[i] = sum_j f[j] * w[j-i] for j >= i
        # conv_fw[j] = sum_i f[i] * w[N-1-j+i]
        # This is getting complicated. Let me just use the direct approach with FFT for the whole thing.
        
        # Alternative: grad_f = 2 * (f * weights) for circular, but we have linear
        # For linear autocorrelation, the gradient is:
        # d/d f[i] sum_k w_k ac_k = 2 * sum_k w_k * (f[i+k] if i+k<N) + sum_k w_k * (f[i-k] if i-k>=0)
        # Wait, ac[k] = sum_j f[j] f[j+k], so d(ac[k])/df[i] = f[i+k] (if i+k<N) + f[i-k] (if i-k>=0)
        # So grad[i] = sum_k w_k * [f[i+k]*1_{i+k<N} + f[i-k]*1_{i-k>=0}]
        
        # Let's compute this using FFT-based convolution
        # Term 1: sum_k w_k * f[i+k] for i+k < N
        # Term 2: sum_k w_k * f[i-k] for i-k >= 0
        
        # Term 2 is simpler: sum_k w_k * f[i-k] = (w * f)[i] (linear convolution)
        # where (w*f)[i] = sum_k w[k] * f[i-k]
        
        # Term 1: sum_k w_k * f[i+k] = sum_k w[N-1-k] * f[i+k] ... no
        # Let j = i+k, so k = j-i: sum_j f[j] * w[j-i] for j >= i and j < N
        # This is the cross-correlation: (f ⋆ w)[i] = sum_j f[j] w[j-i]
        
        # Using FFT for cross-correlation:
        # (f ⋆ w)[i] = ifft(fft(f) * conj(fft(w)))[i] for circular, need linear
        nfft2 = 1
        while nfft2 < 2 * N:
            nfft2 *= 2
        F2 = np.fft.rfft(f, n=nfft2)
        W2 = np.fft.rfft(weights, n=nfft2)
        
        # Cross-correlation term: sum_j f[j] * w[j-i]
        # = ifft(conj(fft(w)) * fft(f)) but this gives circular
        # For linear: pad both to nfft2
        corr = np.fft.irfft(np.conj(W2) * F2, n=nfft2)
        # corr[i] = sum_j f[j] * w[j-i] (circular, but with padding it's linear for i in [0, N-1])
        term1 = corr[:N]
        
        # Convolution term: sum_k w[k] * f[i-k]
        # = ifft(fft(w) * fft(f)) 
        conv = np.fft.irfft(W2 * F2, n=nfft2)
        term2 = conv[:N]
        
        grad_f = term1 + term2
        return grad_f
    
    # Objective with log-sum-exp
    def objective_and_grad(x, beta=100.0):
        """Compute smooth objective and its gradient efficiently."""
        f = np.maximum(x, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10, np.zeros(N)
        
        ac = compute_ac(f)
        ac_safe = np.maximum(ac, 1e-15)
        
        # Log-sum-exp for max approximation
        m = np.max(ac_safe)
        if m < 1e-12:
            return 1e10, np.zeros(N)
        
        exp_ac = np.exp((ac_safe - m) * beta)
        sum_exp = np.sum(exp_ac)
        weights = exp_ac / sum_exp  # Softmax weights
        
        log_sum_exp = m + np.log(sum_exp / beta)
        
        # Gradient of log_sum_exp w.r.t. f
        grad_lse = compute_grad_ac(f, weights)
        
        # Objective: N * log_sum_exp / s^2
        # d(obj)/df[i] = N * (grad_lse[i] * s^2 - log_sum_exp * 2*s * 1) / s^4
        # = N * (grad_lse[i] * s - 2 * log_sum_exp) / s^3
        
        s2 = s * s
        s3 = s2 * s
        obj = N * log_sum_exp / s2
        obj_grad = N * (grad_lse * s - 2.0 * log_sum_exp) / s3
        
        return obj, obj_grad
    
    # Projected gradient descent with adaptive step size
    f_current = flat.copy()
    lr = 0.05
    beta = 50.0
    
    for iteration in range(500):
        if time.time() - t0 > deadline:
            break
        
        obj, grad = objective_and_grad(f_current, beta)
        
        # Gradient step with projection
        f_new = f_current - lr * grad
        f_new = np.maximum(f_new, 0.0)
        
        # Check if improved
        obj_new, _ = objective_and_grad(f_new, beta)
        if obj_new < obj:
            f_current = f_new
            exact_val = autoconv_max_normalized(f_current)
            if exact_val < best_val:
                best_val = exact_val
                best_f = f_current.copy()
        else:
            # Reduce learning rate
            lr *= 0.8
        
        # Increase beta gradually for sharper max approximation
        if iteration % 50 == 0 and beta < 200:
            beta = min(beta * 1.5, 200.0)
    
    # Final evaluation with exact objective
    final_val = autoconv_max_normalized(best_f)
    if final_val < baseline_val:
        return list(best_f)
    
    # Return flat function
    return list(flat)
# EVOLVE-BLOCK-END
