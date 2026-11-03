import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill
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

def _smooth_obj_and_grad(heights, alpha):
    """Compute smooth objective and gradient for L-BFGS-B.
    
    Uses log-sum-exp for smooth max approximation.
    """
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Log-sum-exp for smooth max
    max_c = np.max(conv)
    shifted = conv - max_c
    exp_conv = np.exp(alpha * shifted)
    sum_exp = np.sum(exp_conv)
    smooth_max = max_c + np.log(sum_exp) / alpha
    
    # Gradient weights
    weights = exp_conv / sum_exp
    
    # Gradient of smooth_max w.r.t. heights
    # d(smooth_max)/d(h_i) = sum_j weights[j] * d(conv[j])/d(h_i)
    # d(conv[j])/d(h_i) = h[j-i] + h[j+i] (with zero padding)
    
    # Compute gradient efficiently using FFT
    W = np.fft.rfft(weights)
    H = np.fft.rfft(f)
    
    # Term 1: sum_j weights[j] * h[j-i] = cross-correlation
    term1 = np.fft.irfft(W * np.conj(H), n=M_FFT)
    
    # Term 2: sum_j weights[j] * h[j+i]
    # This is the same as term1 by symmetry for real signals
    # Actually, let's verify: for real signals, the correlation is symmetric
    # term2[i] = sum_j weights[j] * h[j+i]
    # Let k = j+i, then j = k-i
    # term2[i] = sum_k weights[k-i] * h[k]
    # This is the correlation of weights and h at lag -i
    # For real signals, corr[i] = corr[-i] if the signals are symmetric
    # But they're not necessarily symmetric, so we need to compute both
    
    # Actually, for the autoconvolution, we have a special structure.
    # Let me think more carefully.
    # conv[j] = sum_i h[i] * h[j-i]
    # d(conv[j])/d(h[i]) = h[j-i] + h[i-j] (when both indices are valid)
    # Wait, that's not quite right either.
    # conv[j] = sum_k h[k] * h[j-k]
    # d(conv[j])/d(h[i]) = h[j-i] + h[i-j] if 0<=j-i<n and 0<=i-j<n
    # But since h is real, h[j-i] and h[i-j] are different unless j=i
    
    # Let's use a different approach. The gradient of the autoconvolution
    # can be computed as:
    # grad[i] = sum_j weights[j] * (h[j-i] + h[i-j])
    # where h[k] = 0 for k outside [0, n-1]
    
    # First term: sum_j weights[j] * h[j-i]
    # Let k = j-i, then j = k+i
    # = sum_k weights[k+i] * h[k]
    # This is the cross-correlation of weights and h
    
    # Second term: sum_j weights[j] * h[i-j]
    # Let k = i-j, then j = i-k
    # = sum_k weights[i-k] * h[k]
    # This is also a cross-correlation but with different indexing
    
    # Using FFT:
    # Cross-correlation of a and b: (a ⋆ b)[i] = sum_j a[j] * b[j-i]
    # In frequency domain: ifft(fft(a) * conj(fft(b)))
    
    # Term 1: sum_j weights[j] * h[j-i]
    # This is (weights ⋆ h)[i] where we define (a ⋆ b)[i] = sum_j a[j] * b[j-i]
    # In FFT: ifft(fft(weights) * conj(fft(h)))
    
    # Term 2: sum_j weights[j] * h[i-j]
    # Let's rewrite: sum_j weights[j] * h[i-j]
    # Let k = i-j, then j = i-k
    # = sum_k weights[i-k] * h[k]
    # = sum_k h[k] * weights[i-k]
    # This is (h ⋆ weights)[i] where (a ⋆ b)[i] = sum_j a[j] * b[i-j]
    # In FFT: ifft(conj(fft(h)) * fft(weights)) = ifft(fft(weights) * conj(fft(h)))
    
    # So both terms are the same! That makes sense by symmetry.
    # grad[i] = 2 * (weights ⋆ h)[i]
    
    grad = 2.0 * term1[:n]
    
    # Now compute the objective and its gradient
    total = np.sum(heights)
    w = 0.5 / n
    
    if total <= 1e-10:
        return 1e10, np.zeros(n)
    
    # obj = smooth_max * n / (total^2 * w)
    obj = smooth_max * n / (total * total * w)
    
    # d(obj)/d(h_i) = d(smooth_max)/d(h_i) * n / (total^2 * w) 
    #                + smooth_max * n * d(1/(total^2 * w))/d(h_i)
    #                = grad[i] * n / (total^2 * w)
    #                + smooth_max * n * (-2 * total) / (total^4 * w)
    #                = grad[i] * n / (total^2 * w)
    #                - 2 * smooth_max * n / (total^3 * w)
    
    obj_grad = grad * (n / (total * total * w)) - smooth_max * (2.0 * n / (total * total * total * w))
    
    return obj, obj_grad

def _spectral_heights(magnitudes, phases):
    """Generate heights from spectral representation.
    
    magnitudes: non-negative array of shape (M_FFT//2+1,)
    phases: array of shape (M_FFT//2+1,)
    Returns real heights of shape (N,)
    """
    # Construct complex spectrum
    n_fft = M_FFT
    half = n_fft // 2 + 1
    
    F = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    F = magnitudes * np.exp(1j * phases)
    
    # Make it conjugate symmetric for real output
    F_full = np.zeros(n_fft, dtype=np.complex128)
    F_full[:half] = F
    # For even n_fft, the Nyquist frequency (index n_fft//2) is its own conjugate
    # So we need to handle it carefully
    if n_fft % 2 == 0:
        # F_full[n_fft//2] should be real
        F_full[n_fft//2] = np.real(F[half-1])
        F_full[n_fft//2+1:] = np.conj(F_full[1:half][::-1])
    else:
        F_full[half:] = np.conj(F_full[1:half][::-1])
    
    h = np.fft.irfft(F_full, n=n_fft)
    return h[:N]

def _spectral_score(magnitudes, phases):
    """Score for spectral representation."""
    h = _spectral_heights(magnitudes, phases)
    h = np.maximum(h, 0.0)
    if np.sum(h) <= 0:
        return 1e10
    return compute_score(h)

def _spectral_optimize(magnitudes, phases, time_limit):
    """Optimize spectral magnitudes and phases to minimize score."""
    start_time = time.time()
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    # Convert to log-magnitudes for optimization (ensures non-negativity)
    log_mags = np.log(magnitudes + 1e-10)
    
    def objective(params):
        log_mags_opt = params[:len(log_mags)]
        phases_opt = params[len(log_mags):]
        
        mags = np.exp(log_mags_opt)
        h = _spectral_heights(mags, phases_opt)
        h = np.maximum(h, 0.0)
        
        if np.sum(h) <= 0:
            return 1e10
        return compute_score(h)
    
    # Initial result
    h_init = _spectral_heights(magnitudes, phases)
    h_init = np.maximum(h_init, 0.0)
    score_init = compute_score(h_init)
    
    best_h = h_init.copy()
    best_score = score_init
    
    # Use L-BFGS-B to optimize log-magnitudes and phases
    try:
        result = minimize(
            objective,
            np.concatenate([log_mags, phases]),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.x is not None:
            log_mags_opt = result.x[:len(log_mags)]
            phases_opt = result.x[len(log_mags):]
            mags_opt = np.exp(log_mags_opt)
            h_opt = _spectral_heights(mags_opt, phases_opt)
            h_opt = np.maximum(h_opt, 0.0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    except Exception:
        pass
    
    return best_h, best_score

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
    
    # 6. Spectral candidates: random phases with uniform magnitudes
    half = M_FFT // 2 + 1
    for trial in range(10):
        mags = np.ones(half)
        phases = rng.uniform(0, 2*np.pi, half)
        # Make sure the spectrum is conjugate symmetric
        h_spec = _spectral_heights(mags, phases)
        h_spec = np.maximum(h_spec, 0.0)
        if np.sum(h_spec) > 0:
            candidates.append(h_spec)
    
    # 7. Spectral candidates: low-pass filtered
    for cutoff in [0.1, 0.2, 0.3, 0.5]:
        mags = np.zeros(half)
        mags[:int(cutoff * half)] = 1.0
        phases = rng.uniform(0, 2*np.pi, half)
        h_spec = _spectral_heights(mags, phases)
        h_spec = np.maximum(h_spec, 0.0)
        if np.sum(h_spec) > 0:
            candidates.append(h_spec)
    
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
    
    # Strategy 1: Spectral optimization
    # Convert best_h to spectral representation
    if time_left() > 60:
        f_best = np.zeros(M_FFT, dtype=np.float64)
        f_best[:N] = best_h
        F_best = np.fft.rfft(f_best)
        
        mags_best = np.abs(F_best)
        phases_best = np.angle(F_best)
        
        # Optimize spectral representation
        h_spec, score_spec = _spectral_optimize(mags_best, phases_best, time_left() - 10)
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Multi-start Simulated Annealing
    if time_left() > 40:
        num_chains = 3
        steps_per_chain = 1000
        
        for chain_idx in range(num_chains):
            if deadline_reached():
                break
            
            if chain_idx == 0:
                current_h = best_h.copy()
            else:
                if rng.random() < 0.5:
                    current_h = candidates[rng.integers(0, len(candidates))].copy()
                else:
                    current_h = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
            
            current_score = compute_score(current_h)
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            
            T_start = 0.1
            T_end = 0.001
            step_size = 0.1
            
            for step in range(steps_per_chain):
                if deadline_reached():
                    break
                
                progress = step / steps_per_chain
                T = T_start * (T_end / T_start) ** progress
                
                noise = rng.normal(0, step_size, N)
                new_h = current_h + noise
                new_h = np.maximum(new_h, 0.0)
                
                if np.sum(new_h) <= 0:
                    continue
                    
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    accept = True
                else:
                    delta = new_score - current_score
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
    
    # Strategy 3: Gradient-based optimization with smooth surrogate (L-BFGS-B)
    if time_left() > 25:
        alpha_schedule = [20.0, 50.0, 100.0, 200.0]
        
        for alpha_val in alpha_schedule:
            if deadline_reached():
                break
            
            h_opt = best_h.copy()
            
            def objective_wrapper(h, alpha=alpha_val):
                obj, _ = _smooth_obj_and_grad(h, alpha)
                return obj
            
            def gradient_wrapper(h, alpha=alpha_val):
                _, grad = _smooth_obj_and_grad(h, alpha)
                return grad
            
            try:
                result = minimize(
                    objective_wrapper,
                    h_opt,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.x is not None and np.all(result.x >= -1e-10):
                    h_new = np.maximum(result.x, 0.0)
                    score = compute_score(h_new)
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
            except Exception:
                pass
            
            # Random restarts
            for restart in range(2):
                if deadline_reached():
                    break
                
                h_init = np.maximum(np.ones(N) + rng.normal(0, 0.3, N), 0.0)
                
                try:
                    result = minimize(
                        objective_wrapper,
                        h_init,
                        jac=gradient_wrapper,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.x is not None and np.all(result.x >= -1e-10):
                        h_new = np.maximum(result.x, 0.0)
                        score = compute_score(h_new)
                        if score < best_score:
                            best_score = score
                            best_h = h_new.copy()
                except Exception:
                    pass
    
    # Final polish: Local random search with adaptive step size
    if time_left() > 10:
        polish_steps = 150
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
