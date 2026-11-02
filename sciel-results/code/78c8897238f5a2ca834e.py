import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    return conv

def _smooth_max_and_grad(heights, beta):
    """Compute smooth max of autoconvolution via log-sum-exp and its gradient.
    
    Args:
        heights: array of shape (N,)
        beta: temperature parameter (larger = sharper max approximation)
    
    Returns:
        smooth_max: smooth approximation of max(conv)
        grad: gradient of smooth_max w.r.t. heights
    """
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Log-sum-exp for smooth max
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    smooth_max = max_c + np.log(np.sum(weights)) / beta
    
    # Gradient: d(smooth_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i) / beta
    # conv_j = sum_k h_k * h_{j-k} (with appropriate indexing)
    # d(conv_j)/d(h_i) = h_{j-i} + h_{j+i} (for linear convolution with zero padding)
    # 
    # More precisely: conv[j] = sum_{k=0}^{n-1} h[k] * h[j-k] where h[m]=0 for m<0 or m>=n
    # d(conv[j])/d(h[i]) = h[j-i] + h[j+i] (the two terms where i appears)
    
    # We need: grad[i] = sum_j weights[j] * (h[j-i] + h[j+i]) / beta
    # This can be computed as: grad = (weights * h) convolved appropriately
    
    # Let's compute this more carefully using the structure
    # conv[j] = sum_k h[k]*h[j-k]
    # d(conv[j])/d(h[i]) = h[j-i] + h[j+i] (when indices are valid)
    
    # grad[i] = (1/beta) * sum_j w[j] * (h[j-i] + h[j+i])
    #          = (1/beta) * (sum_j w[j]*h[j-i] + sum_j w[j]*h[j+i])
    #          = (1/beta) * ((w * h)[i] + (w * h)[i])  [by symmetry of convolution]
    # Wait, let me be more careful.
    
    # Let A[j] = sum_k h[k]*h[j-k] = conv[j]
    # dA[j]/dh[i] = h[j-i] + h[j+i]
    
    # grad[i] = (1/beta) * sum_j w[j] * (h[j-i] + h[j+i])
    #         = (1/beta) * [sum_j w[j]*h[j-i] + sum_j w[j]*h[j+i]]
    
    # First term: sum_j w[j]*h[j-i] = (w * h)[i] where * is cross-correlation
    # Second term: sum_j w[j]*h[j+i] = (w * h)[-i] = (h * w)[i] by symmetry
    
    # Actually, let's just compute it directly using FFT
    # grad[i] = (1/beta) * sum_j w[j] * (h[j-i] + h[j+i])
    
    # This is: (1/beta) * (conv_w_h[i] + conv_w_h[i]) where conv_w_h[i] = sum_j w[j]*h[j-i]
    # But we need to be careful about the range of j.
    
    # Simpler approach: use the fact that the gradient of sum_j w[j]*conv[j] w.r.t. h[i]
    # is sum_j w[j]*(h[j-i] + h[j+i])
    
    # Let's compute this using FFT:
    # Let W = fft(w) and H = fft(h)
    # The sum sum_j w[j]*h[j-i] is a correlation, which can be done via FFT
    
    # Actually, let's just do it with a direct computation using the structure
    # Since N=600 and M_FFT=2048, we can afford a somewhat direct approach
    
    # Compute the gradient using the formula:
    # grad[i] = (1/beta) * sum_j w[j] * (h[j-i] + h[j+i])
    
    # We can compute this as:
    # term1[i] = sum_j w[j] * h[j-i]  (correlation of w and h)
    # term2[i] = sum_j w[j] * h[j+i]  (this is correlation of w and reversed h, shifted)
    
    # Let's use FFT for efficiency
    w_ext = np.zeros(M_FFT, dtype=np.float64)
    w_ext[:M_FFT] = weights  # weights is already length M_FFT
    h_ext = np.zeros(M_FFT, dtype=np.float64)
    h_ext[:n] = heights
    
    W = np.fft.rfft(w_ext)
    H = np.fft.rfft(h_ext)
    
    # Correlation: sum_j w[j]*h[j-i] = ifft(W * conj(H))[i]
    # But we need to handle the indexing carefully.
    
    # Actually, let's use a simpler approach: just compute the gradient numerically
    # or use a different formulation.
    
    # Alternative: use the fact that grad = (2/beta) * ifft(W * H) restricted to first n
    # Let's verify: if conv = ifft(F*F) where F=fft(h), then
    # d(conv)/dh = 2*ifft(F * fft(dh)) for small dh
    # So d(conv[j])/dh[i] for the full vector is related to ifft(F * something)
    
    # Actually, the cleanest way: 
    # L = sum_j w[j] * conv[j]
    # dL/dh[i] = sum_j w[j] * d(conv[j])/dh[i]
    # d(conv[j])/dh[i] = h[j-i] + h[j+i]
    
    # So dL/dh[i] = sum_j w[j]*(h[j-i] + h[j+i])
    #             = (w ⊛ h)[i] + (w ⊛ h)[i]  where ⊛ denotes appropriate cross-correlation
    #             = 2 * (w ⊛ h)[i]  by symmetry? Not quite.
    
    # Let's just compute it directly for now since N=600 is small enough
    # grad = np.zeros(n)
    # for i in range(n):
    #     for j in range(M_FFT):
    #         if 0 <= j-i < n:
    #             grad[i] += weights[j] * heights[j-i]
    #         if 0 <= j+i < n:
    #             grad[i] += weights[j] * heights[j+i]
    # grad /= beta
    
    # But that's O(N*M) = 600*2048 = 1.2M operations per gradient, which is fine.
    # Actually, let's use FFT properly.
    
    # The sum sum_j w[j]*h[j-i] for i in range(n) can be computed as:
    # Create an array where we place w and h appropriately and use FFT correlation.
    
    # Let's use scipy.signal.correlate if available, or just do it with numpy
    # For now, let's use a straightforward FFT-based approach:
    
    # term1[i] = sum_j w[j] * h[j-i]
    # This is the correlation of w and h at lag i.
    # Using FFT: corr = ifft(fft(w) * conj(fft(h)))
    # But we need to be careful about the definition.
    
    # Let's just compute it directly since it's not too slow
    grad = np.zeros(n)
    
    # Precompute the valid ranges
    for j in range(M_FFT):
        wj = weights[j]
        if wj < 1e-15:
            continue
        # h[j-i] for i in range(n): j-i in [0, n-1] => i in [j-n+1, j]
        i_min = max(0, j - n + 1)
        i_max = min(n - 1, j)
        if i_min <= i_max:
            # h[j-i] for i from i_min to i_max
            idx = j - np.arange(i_min, i_max + 1)
            grad[i_min:i_max+1] += wj * heights[idx]
        
        # h[j+i] for i in range(n): j+i in [0, n-1] => i in [-j, n-1-j]
        i_min2 = max(0, -j)
        i_max2 = min(n - 1, n - 1 - j)
        if i_min2 <= i_max2:
            idx2 = j + np.arange(i_min2, i_max2 + 1)
            grad[i_min2:i_max2+1] += wj * heights[idx2]
    
    grad /= beta
    
    return smooth_max, grad

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered
    x = np.linspace(0, 1, N)
    for width in [0.1, 0.2, 0.3, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks
    for left_frac in [0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.1, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Multi-Start Simulated Annealing directly on 600D heights
    # We will run several independent SA chains
    
    num_chains = 3
    steps_per_chain = 1500
    
    for chain_idx in range(num_chains):
        if deadline_reached():
            break
        
        # Initialize this chain from a random candidate or best
        if chain_idx == 0:
            current_h = best_h.copy()
        else:
            # Pick a random candidate from the list, or generate new random
            if rng.random() < 0.5:
                current_h = candidates[rng.integers(0, len(candidates))].copy()
            else:
                # Random Gaussian perturbation around uniform
                current_h = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
            
            # Normalize to similar scale? Not strictly necessary if score is scale invariant?
            # Score = max_conv * N / (total^2 * w).
            # If h -> c*h, max_conv -> c^2 max_conv, total -> c total.
            # Score -> c^2 / c^2 = 1. Scale invariant.
            # So we don't need to normalize, but non-negativity is crucial.
            
        current_score = compute_score(current_h)
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # SA Parameters
        T_start = 0.1
        T_end = 0.001
        # Adaptive step size
        step_size = 0.1
        
        for step in range(steps_per_chain):
            if deadline_reached():
                break
                
            # Annealing schedule
            progress = step / steps_per_chain
            T = T_start * (T_end / T_start) ** progress
            
            # Generate perturbation
            # Add Gaussian noise
            noise = rng.normal(0, step_size, N)
            new_h = current_h + noise
            new_h = np.maximum(new_h, 0.0)
            
            if np.sum(new_h) <= 0:
                continue
                
            new_score = compute_score(new_h)
            
            # Acceptance
            if new_score < current_score:
                accept = True
            else:
                delta = new_score - current_score
                # Only accept small deteriorations
                if delta < 0.05: 
                    accept = rng.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                
                step_size = max(0.01, step_size * 0.999)
            else:
                step_size = max(0.01, step_size * 0.95)
                
    # Gradient-based refinement using smooth surrogate
    if time_left() > 15:
        # Use log-sum-exp smooth max with increasing beta
        current_h = best_h.copy()
        current_score = best_score
        
        # Start with moderate beta and increase
        beta_values = [10, 50, 100, 200, 500, 1000]
        
        for beta in beta_values:
            if deadline_reached():
                break
            
            # Run gradient descent for several steps
            lr = 0.01 / beta  # Learning rate scaled by beta
            max_iters = 200
            no_improve = 0
            
            for it in range(max_iters):
                if deadline_reached():
                    break
                
                smooth_max, grad = _smooth_max_and_grad(current_h, beta)
                
                # Normalize gradient by the scale of the problem
                # The score is scale invariant, so we should normalize
                total = np.sum(current_h)
                if total <= 0:
                    break
                
                # Normalize gradient to have unit norm for stable updates
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break
                grad = grad / grad_norm
                
                # Project gradient onto feasible direction (non-negative)
                # Simple approach: take step and clip
                new_h = current_h - lr * grad
                new_h = np.maximum(new_h, 0.0)
                
                new_score = compute_score(new_h)
                
                if new_score < current_score - 1e-12:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    no_improve = 0
                    # Decrease learning rate slightly
                    lr *= 0.99
                else:
                    no_improve += 1
                    lr *= 0.9
                    if no_improve > 50:
                        break
    
    # Final polish: Local random search around best_h
    if time_left() > 10:
        polish_steps = 300
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.05
        
        for _ in range(polish_steps):
            if deadline_reached():
                break
            noise = rng.normal(0, step_size, N)
            new_h = np.maximum(current_h + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.1, step_size * 1.05)
            else:
                step_size = max(0.005, step_size * 0.9)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
