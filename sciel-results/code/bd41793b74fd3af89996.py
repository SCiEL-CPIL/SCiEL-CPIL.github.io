import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Start with uniform distribution
    h = np.ones(N)
    
    # Compute autocorrelation
    def autoconvolve(h):
        # Full autocorrelation via FFT
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        ac = np.fft.irfft(H * np.conj(H), n=nfft)
        # Take the relevant portion (lags from -(n-1) to (n-1))
        # The full autocorrelation has 2*n-1 elements
        # We want lags 0 to n-1 (the positive side)
        # Actually for minimization of max, we need the full range
        return ac[:2*n-1]
    
    def score(h):
        ac = autoconvolve(h)
        # Normalize by sum of h (L1 norm)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        return np.max(ac) / l1
    
    # Current best
    best_h = h.copy()
    best_score = score(h)
    
    # Use gradient-based optimization with log-sum-exp soft-max
    # Objective: minimize max(ac) / sum(h)
    # Use soft-max: (1/tau) * log(sum(exp(tau * ac))) as smooth surrogate for max(ac)
    
    # Gradient descent with momentum
    learning_rate = 0.01
    momentum = 0.9
    v = np.zeros_like(h)
    tau = 50.0  # soft-max temperature
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute soft-max objective and gradient
        ac = autoconvolve(h)
        l1 = np.sum(h)
        if l1 < 1e-10:
            break
        
        # Soft-max approximation of max(ac)
        tau_ac = tau
        max_ac = np.max(ac)
        exp_ac = np.exp(tau_ac * (ac - max_ac))
        log_sum_exp = max_ac + np.log(np.sum(exp_ac)) / tau_ac
        
        # Gradient of soft-max wrt ac
        # d(log_sum_exp)/d(ac_i) = exp(tau*ac_i) / sum(exp(tau*ac_j))
        grad_ac = exp_ac / np.sum(exp_ac)
        
        # Gradient of ac wrt h
        # ac = h * h (autocorrelation)
        # d(ac_k)/d(h_i) = h_{i+k} + h_{i-k} (with appropriate bounds)
        # This is complex, so use numerical gradient or FFT-based
        
        # For efficiency, use the fact that ac is computed via FFT
        # Let's use a simpler approach: perturb and measure
        
        # Actually, let's use a different approach: direct optimization
        # with projected gradient descent
        
        # Compute gradient numerically for a few random directions
        # Or use the analytical gradient
        
        # The autocorrelation is: ac[k] = sum_i h[i] * h[i+k]
        # d(ac[k])/d(h[j]) = h[j+k] + h[j-k] (with bounds checking)
        
        # Gradient of objective = (1/l1) * sum_k grad_ac[k] * d(ac[k])/d(h)
        # minus (max_ac / l1^2) * sum(h) derivative
        
        # This is getting complex. Let's use a simpler heuristic:
        # Try to reduce the peak by spreading mass more evenly
        
        # Use a simple iterative approach: reduce high values, increase low values
        # while keeping sum constant
        
        if iteration % 100 == 0:
            # Normalize to keep L1 = 1
            h = h / np.sum(h)
        
        # Compute gradient via finite differences on a small set of coordinates
        # For speed, use analytical gradient
        
        # d(ac)/d(h) can be computed as:
        # If ac = ifft(fft(h) * conj(fft(h))), then d(ac)/d(h) is related to h
        # Specifically, ac[k] = sum_i h[i]*h[i+k]
        # d(ac[k])/d(h[j]) = h[j+k] + h[j-k] (with proper indexing)
        
        # Let's compute the gradient properly
        n = len(h)
        grad_h = np.zeros(n)
        
        # For each lag k, ac[k] = sum_i h[i]*h[i+k]
        # d(ac[k])/d(h[j]) = h[j+k] if j+k < n, plus h[j-k] if j-k >= 0
        # But we need to sum over all k with weights grad_ac[k]
        
        # This is equivalent to: grad_h[j] = sum_k grad_ac[k] * (h[j+k] + h[j-k])
        # which is a convolution-like operation
        
        # Use FFT for efficiency
        # grad_h = ifft(fft(grad_ac) * conj(fft(h))) + ifft(fft(grad_ac) * fft(h))
        # Actually, it's: grad_h = ifft(fft(h) * fft(grad_ac)) + ifft(fft(h) * fft(grad_ac_flipped))
        
        # Let's just do it directly for now (N=600 is small)
        # grad_h[j] = sum_k grad_ac[k] * h[j+k] (for j+k < n) + sum_k grad_ac[k] * h[j-k] (for j-k >= 0)
        
        # Vectorized approach:
        # For each j, grad_h[j] = sum_{k: j+k<n} grad_ac[k]*h[j+k] + sum_{k: j-k>=0} grad_ac[k]*h[j-k]
        
        # This is: grad_h = convolve(h, grad_ac, 'valid') + convolve(h, grad_ac[::-1], 'valid')
        # But we need to be careful with the exact convolution type
        
        # Simpler: use the fact that this is a cross-correlation
        # grad_h[j] = sum_k grad_ac[k] * h[j+k] + sum_k grad_ac[k] * h[j-k]
        
        # Let's compute it with np.correlate
        # term1[j] = sum_k grad_ac[k] * h[j+k] = correlate(h, grad_ac)[j + n - 1] (approximately)
        # This is getting complicated. Let's use a simpler update rule.
        
        # Heuristic: reduce peaks in h where they contribute to autocorrelation peaks
        # and increase valleys
        
        # Simple approach: gradient step with projection
        # Approximate gradient as: where ac is high, reduce h at positions that contribute
        # For simplicity, use: grad_h ≈ h * (ac[0] - mean(ac))
        # This encourages reducing h where the local autocorrelation contribution is high
        
        # Better: use the actual gradient via FFT
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        G = np.fft.rfft(grad_ac, n=nfft)
        
        # d(ac)/d(h) in frequency domain
        # ac = ifft(H * conj(H))
        # d(ac)/d(h) = ifft(2 * H * conj(dH/dh)) where dH/dh is the derivative of FFT
        # This is tricky. Let's use numerical gradient for a few iterations
        
        # For now, use a simple heuristic update
        # Reduce h at positions where the local density is high relative to neighbors
        # and where the autocorrelation peak is high
        
        # Normalize h to sum = 1
        h = h / np.sum(h)
        
        # Compute local contribution to autocorrelation peak
        # ac[0] = sum h[i]^2
        # To reduce ac[0], we want to reduce h[i]^2, i.e., make h more uniform
        # Gradient of ac[0] wrt h[i] = 2*h[i]
        # So gradient step: h[i] -= lr * 2*h[i] * grad_ac[0]
        
        # But we also need to account for the normalization constraint
        # Use Lagrange multiplier or projection
        
        # Simple update: h[i] = h[i] - lr * 2 * h[i] * grad_ac[0] / l1
        # Then project to sum = 1
        
        lr = 0.001
        h_new = h - lr * 2 * h * grad_ac[0] / l1
        
        # Ensure non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Normalize to sum = 1
        if np.sum(h_new) > 1e-10:
            h_new = h_new / np.sum(h_new)
        else:
            h_new = h
        
        # Check if improved
        new_score = score(h_new)
        if new_score < best_score:
            best_h = h_new.copy()
            best_score = new_score
            h = h_new
        
        # Momentum update (simplified)
        v = momentum * v + (1 - momentum) * (h_new - h)
        h = h_new
    
    # Final polish: try a few random perturbations around best_h
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline:
                break
            perturbation = np.random.randn(N) * 0.01
            h_pert = best_h + perturbation
            h_pert = np.maximum(h_pert, 0)
            if np.sum(h_pert) > 1e-10:
                h_pert = h_pert / np.sum(h_pert)
            s = score(h_pert)
            if s < best_score:
                best_h = h_pert
                best_score = s
    
    return list(best_h)
# EVOLVE-BLOCK-END
