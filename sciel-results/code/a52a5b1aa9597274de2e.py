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

def _score_and_grad(heights):
    """Compute score and gradient of score w.r.t. heights using FFT.
    The score is scale-invariant, so we normalize heights to have sum=1 for gradient computation.
    Returns (score, grad) where grad is the gradient of score w.r.t. heights (before normalization).
    """
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf, np.zeros(n)
    
    # Normalize to sum=1 for scale-invariant computation
    h_norm = heights / total
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h_norm
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    
    # Score for normalized heights: max_conv * n / (1 * 1 * w) = max_conv * n / w
    # But original score = max_conv_orig * n / (total^2 * w)
    # max_conv_orig = total^2 * max_conv (since h_orig = total * h_norm)
    # So score = total^2 * max_conv * n / (total^2 * w) = max_conv * n / w
    score = max_conv * n / w
    
    # Gradient: We need d(score)/d(heights)
    # score = max_conv * n / w where max_conv = max_k (h_norm * h_norm)[k]
    # Let k* be the argmax
    # d(max_conv)/d(h_norm[i]) = 2 * (h_norm * delta_{k*})[i] where delta_{k*} is the indicator of position k*
    # Actually, conv[i] = sum_j h_norm[j] * h_norm[i-j]
    # d(conv[k*])/d(h_norm[i]) = h_norm[i-k*] + h_norm[k*-i] (with appropriate indexing)
    
    # For linear convolution: conv[k] = sum_j h[j] * h[k-j]
    # d(conv[k*])/d(h[i]) = h[k*-i] + h[i-k*] (if k*-i >= 0 and i-k* >= 0)
    
    # Since h is defined on [0, n-1], and conv is on [0, 2n-2]
    # Let's compute the gradient properly
    
    # The convolution at lag k* is: sum_{j=max(0,k*-n+1)}^{min(n-1, k*)} h_norm[j] * h_norm[k*-j]
    # Gradient w.r.t. h_norm[i]:
    # If 0 <= i < n:
    #   term1: h_norm[k*-i] if 0 <= k*-i < n
    #   term2: h_norm[i-k*] if 0 <= i-k* < n
    
    k_star = np.argmax(conv)
    grad_conv = np.zeros(n)
    
    # h_norm[k*-i] for i in [0, n-1]
    idx1 = k_star - np.arange(n)
    valid1 = (idx1 >= 0) & (idx1 < n)
    grad_conv[valid1] += h_norm[idx1[valid1]]
    
    # h_norm[i-k*] for i in [0, n-1]
    idx2 = np.arange(n) - k_star
    valid2 = (idx2 >= 0) & (idx2 < n)
    grad_conv[valid2] += h_norm[idx2[valid2]]
    
    # d(score)/d(h_norm) = (n/w) * grad_conv
    grad_norm = (n / w) * grad_conv
    
    # Chain rule: h_norm = heights / total
    # d(h_norm[i])/d(heights[j]) = delta_{ij}/total - heights[i]*heights[j]/total^2
    # So d(score)/d(heights[j]) = sum_i d(score)/d(h_norm[i]) * (delta_{ij}/total - heights[i]*heights[j]/total^2)
    # = grad_norm[j]/total - heights[j]/total^2 * sum_i grad_norm[i] * heights[i]
    
    sum_grad_h = np.dot(grad_norm, heights)
    grad = grad_norm / total - heights / (total * total) * sum_grad_h
    
    return score, grad

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
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
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
    for _ in range(10):
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

    # 6. Smooth bumps
    for center in [0.2, 0.3, 0.5, 0.7, 0.8]:
        for width in [0.05, 0.1, 0.2]:
            hb = np.exp(-((x - center) / width) ** 2)
            candidates.append(hb)

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

    # Multi-Start Simulated Annealing with gradient-assisted moves
    # We will run several independent SA chains
    
    num_chains = 8
    steps_per_chain = 3000
    
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
            # Mix of random noise and gradient-based moves
            if rng.random() < 0.3 and step > 100:
                # Gradient-based move
                _, grad = _score_and_grad(current_h)
                # Normalize gradient
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    grad = grad / grad_norm
                    # Move in negative gradient direction
                    new_h = current_h - step_size * grad
                    new_h = np.maximum(new_h, 0.0)
                else:
                    noise = rng.normal(0, step_size, N)
                    new_h = current_h + noise
                    new_h = np.maximum(new_h, 0.0)
            else:
                # Random Gaussian noise
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
                
    # Final polish: Gradient-based local optimization around best_h
    if time_left() > 10:
        current_h = best_h.copy()
        current_score = best_score
        
        # Gradient descent with line search
        for _ in range(2000):
            if deadline_reached():
                break
            
            score, grad = _score_and_grad(current_h)
            if score < best_score:
                best_score = score
                best_h = current_h.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            # Normalize gradient
            grad = grad / grad_norm
            
            # Backtracking line search
            step_size = 0.1
            for _ in range(20):
                new_h = current_h - step_size * grad
                new_h = np.maximum(new_h, 0.0)
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    break
                else:
                    step_size *= 0.5
            
            # If we couldn't find a descent step, try random perturbation
            if step_size < 1e-6:
                noise = rng.normal(0, 0.01, N)
                new_h = np.maximum(current_h + noise, 0.0)
                new_score = compute_score(new_h)
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
