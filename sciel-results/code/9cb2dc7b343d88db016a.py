import numpy as np
import time

N = 100
W = 2.0 / N

def _project_sum_box(h, target_sum):
    """Project h onto {0 <= x_i <= 1, sum(x) = target_sum} using bisection on threshold."""
    # We want to find tau such that sum(clip(h - tau, 0, 1)) = target_sum
    # f(tau) = sum(clip(h - tau, 0, 1)) is decreasing in tau
    lo, hi = h.min() - 2.0, h.max() + 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h - mid, 0.0, 1.0))
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2.0
    return np.clip(h - tau, 0.0, 1.0)

def _smooth_overlap(h, alpha):
    """Compute log-sum-exp of overlaps with parameter alpha, plus analytic gradient."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    
    # Log-sum-exp for numerical stability
    m = np.max(overlaps)
    if m > 1e10:
        return m, np.zeros_like(h)
    
    # LSE = m + log(sum(exp(overlaps - m)))
    exp_vals = np.exp(overlaps - m)
    lse = m + np.log(np.sum(exp_vals))
    
    # Gradient of LSE w.r.t h
    # d LSE / d h[i] = sum_k (d overlap_k / d h[i]) * p_k
    # where p_k = exp(overlap_k - m) / sum(exp(overlaps - m))
    #
    # overlap_k = W * sum_j h[j] * (1 - h[(j+k)%N])
    # d overlap_k / d h[i] = W * [ (1 - h[(i+k)%N]) - h[(i-k)%N] ]
    #
    # grad[i] = sum_k p_k * W * [ (1 - h[(i+k)%N]) - h[(i-k)%N] ]
    #
    # This can be computed via circular convolution:
    # term1[i] = sum_k p_k * (1 - h[(i+k)%N]) = sum_k p_k - sum_k p_k * h[(i+k)%N]
    #          = 1 - circ_corr(p, h)[i]  (since sum p_k = 1)
    # term2[i] = sum_k p_k * h[(i-k)%N] = circ_corr(p, h)[i]
    #
    # Wait, let's be precise about circular correlation.
    # Let P = fft(p), Hf = fft(h)
    # circ_corr[p, h][i] = sum_k p[k] * h[(i-k)%N]  (this is what ifft(P * conj(Hf)) gives? No.)
    #
    # Standard circular cross-correlation: (p ⊛ h)[i] = sum_k p[k] h[(i-k) mod N]
    # FFT: ifft(fft(p) * conj(fft(h))) gives sum_k p[k] h[(i-k) mod N]? 
    # Actually, ifft(fft(p) * fft(h)[::-1])... let's just use the property:
    # ifft(fft(p) * conj(fft(h)))_i = sum_j p_j h_{(i-j) mod N}
    # Yes.
    
    P = np.fft.fft(overlaps - m) # Wait, p_k is proportional to exp(overlaps_k - m)
    # p_k = exp_vals[k] / sum(exp_vals)
    p = exp_vals / np.sum(exp_vals)
    
    P = np.fft.fft(p)
    Hf = np.fft.fft(h)
    # corr[i] = sum_k p[k] * h[(i-k)%N]
    corr = np.fft.ifft(P * np.conj(Hf)).real
    
    # term1[i] = sum_k p[k] * (1 - h[(i+k)%N])
    # Note: h[(i+k)%N] is h at index (i+k). Let j = -k. Then sum_k p[k] h[(i+k)%N] = sum_j p[-j] h[(i-j)%N].
    # This is the cross-correlation with reversed p.
    # Let's compute directly using FFT for term1 and term2.
    
    # term2[i] = sum_k p[k] * h[(i-k)%N] = corr[i]
    # term1[i] = sum_k p[k] * (1 - h[(i+k)%N]) = 1 - sum_k p[k] h[(i+k)%N]
    # Let q[k] = p[-k mod N]. Then sum_k p[k] h[(i+k)%N] = sum_k q[-k] h[(i+k)%N] ... this is getting confusing.
    # Easier: 
    # sum_k p[k] h[(i+k)%N] is the cross-correlation of p and h, but with p reversed?
    # Actually, if we define r[k] = p[(N-k)%N], then sum_k p[k] h[(i+k)%N] = sum_k r[-k] h[(i+k)%N] ...
    # Let's just compute it via FFT:
    # ifft(fft(p) * fft(h)[::-1]) ... no.
    # 
    # Let's stick to the definition:
    # A[i] = sum_k p[k] * h[(i+k)%N]
    # This is the circular cross-correlation of p and h, specifically (p ⋆ h)[i].
    # In FFT: ifft(fft(p) * conj(fft(h))) gives B[i] = sum_k p[k] h[(i-k)%N].
    # We want A[i] = sum_k p[k] h[(i+k)%N].
    # A[i] = sum_k p[k] h[(i-(-k))%N]. Let j = -k. A[i] = sum_j p[-j] h[(i-j)%N].
    # So A is the correlation of p_reversed and h.
    # p_reversed[k] = p[(N-k)%N].
    
    p_rev = p[::-1].copy()
    p_rev[0] = p[0] # p[0] is its own reverse in circular sense? No, p_rev[k] = p[-k mod N].
    # p_rev = np.roll(p, 1)? No.
    # p_rev[k] = p[(N-k)%N].
    # p_rev = p[(np.arange(N) * -1) % N]
    p_rev = p[(-np.arange(N)) % N]
    
    P_rev = np.fft.fft(p_rev)
    A = np.fft.ifft(P_rev * np.conj(Hf)).real
    
    # grad[i] = W * ( (1 - A[i]) - corr[i] )
    grad = W * ((1.0 - A) - corr)
    
    # Scale by alpha for the smooth objective
    return alpha * lse, alpha * grad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: uniform
    h = np.full(N, 0.5)
    
    # Parameters
    alpha = 20.0  # Log-sum-exp sharpness
    lr = 0.01     # Initial learning rate
    momentum = 0.9
    v = np.zeros(N)
    
    best_h = h.copy()
    best_val = _smooth_overlap(h, alpha)[0]
    
    # Main optimization loop
    for step in range(5000):
        if time.time() > deadline:
            break
        
        val, grad = _smooth_overlap(h, alpha)
        
        # Update momentum
        v = momentum * v - lr * grad
        h_new = h + v
        
        # Project to feasible set: sum = N/2, 0 <= h <= 1
        h_new = _project_sum_box(h_new, N / 2.0)
        
        # Check if improved
        val_new = _smooth_overlap(h_new, alpha)[0]
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate: decay if stuck
        if step % 500 == 0 and step > 0:
            lr *= 0.9
        
        # Increase alpha over time to sharpen the max approximation
        if step % 500 == 0 and alpha < 100.0:
            alpha *= 1.5
    
    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # One final bisection projection to ensure exact sum after clipping
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
