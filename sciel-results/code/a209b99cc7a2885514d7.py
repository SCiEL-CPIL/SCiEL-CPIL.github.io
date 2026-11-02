import time
import numpy as np

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

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
    # Zero-pad to at least 2*n-1 for linear convolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalization: for uniform heights=1, 
    # integral = n*w = 0.5
    # autoconvolution peak = n*w^2 (for a rectangle of width 0.5, peak of triangle is 0.25)
    # Actually for step function with n intervals of width w, all height 1:
    # The autoconvolution at lag 0 is sum(h_i^2 * w) = n * 1 * w = n*w = 0.5
    # Wait, let me think more carefully.
    # 
    # The step function f(x) = h_i for x in [i*w - 0.25, (i+1)*w - 0.25)
    # Autoconvolution: (f*f)(t) = integral f(x) f(t-x) dx
    # At t=0: sum_i h_i^2 * w = n * 1 * w = n*w = 0.5
    # 
    # Normalized max = max(f*f) / (integral(f))^2 * support_length
    # For uniform: 0.5 / (0.5)^2 * 0.5 = 0.5/0.25*0.5 = 1.0
    # 
    # So: score = max_conv / (total * w)^2 * (n * w)
    #       = max_conv * n / (total^2 * w)  ... let me verify
    # integral = total * w
    # score = max_conv / (total*w)^2 * (n*w) = max_conv * n * w / (total^2 * w^2) = max_conv * n / (total^2 * w)
    # For uniform: max_conv = n*w (from discrete sum), total = n
    # score = n*w * n / (n^2 * w) = 1.0 ✓
    
    score = max_conv * n / (total * total * w)
    return score

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    return m + np.log(np.sum(exp_conv)) / alpha

def _smooth_obj(heights, alpha, M):
    """Smooth objective: smooth_max(autoconv) / (sum(heights)*w)^2 * n*w"""
    n = len(heights)
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    sm = _smooth_max(conv, alpha)
    total = np.sum(heights)
    w = 0.5 / n
    if total <= 1e-10:
        return 1e10, None, None
    val = sm / (total * w) ** 2 * (n * w)
    return val, conv, None

def _smooth_grad(heights, alpha, M):
    """Gradient of the smooth objective using FFT-based correlation."""
    n = len(heights)
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    # Gradient of smooth_max via log-sum-exp
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    weights = exp_conv / np.sum(exp_conv)  # softmax weights
    
    # grad_f[i] = sum_k weights[k] * d(conv_k)/d(f_i)
    # d(conv_k)/d(f_i) = 2*f[k-i] if 0 <= k-i < n, else 0
    # So grad_f[i] = 2 * sum_k weights[k] * f[k-i] (where k-i in [0,n))
    # This is 2 * (linear cross-correlation of weights and f) at lag i
    
    # Use FFT for efficient correlation
    W_full = np.fft.rfft(weights)
    F_full = np.fft.rfft(f)
    # Circular correlation in M points: corr[i] = sum_k weights[k] * f[(k-i) mod M]
    # Since f is zero-padded to M >= 2n-1, for i in [0, n-1], 
    # the circular correlation equals linear correlation
    corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
    grad_f = 2.0 * corr[:n]
    
    # Chain rule for normalized objective
    S = np.sum(heights)
    w = 0.5 / n
    if S <= 1e-10:
        return np.zeros(n)
    
    sm = _smooth_max(conv, alpha)
    # obj = sm * n / (S^2 * w)
    # d(obj)/d(f_i) = grad_f[i] * n/(S^2*w) - sm * 2*n/(S^3*w)
    obj_grad = grad_f * (n / (S * S * w)) - sm * (2.0 * n / (S * S * S * w))
    
    return obj_grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Quick structured candidates
    structured_candidates = []
    
    # Tapered edges (cosine taper)
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        h_taper = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h_taper[:taper_len] = ramp
            h_taper[-taper_len:] = ramp[::-1]
        structured_candidates.append(h_taper)
    
    # Raised cosine (smooth bump)
    x = np.linspace(0, 1, N)
    for center in [0.5]:
        for width in [0.3, 0.5, 0.7, 0.9, 1.0]:
            h_cos = np.exp(-((x - center) / width) ** 2)
            h_cos = h_cos / np.max(h_cos)  # normalize to max 1
            structured_candidates.append(h_cos)
    
    # Parabolic
    for a in [0.5, 1.0, 2.0, 3.0]:
        h_par = 1.0 - a * (2 * x - 1) ** 2
        h_par = np.maximum(h_par, 0.0)
        if np.max(h_par) > 0:
            structured_candidates.append(h_par)
    
    # Linear taper
    for slope in [0.5, 1.0, 2.0, 3.0, 5.0]:
        h_lin = 1.0 - slope * np.abs(2 * x - 1)
        h_lin = np.maximum(h_lin, 0.0)
        if np.max(h_lin) > 0:
            structured_candidates.append(h_lin)
    
    # Plateau with smooth edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        h_plat = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h_plat[:edge_len] = ramp
            h_plat[-edge_len:] = ramp[::-1]
        structured_candidates.append(h_plat)
    
    # Evaluate structured candidates
    for h_cand in structured_candidates:
        if time_left() < 5:
            break
        h_cand = np.maximum(h_cand, 0.0)
        if np.sum(h_cand) <= 0:
            continue
        score = compute_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Phase 2: Multi-start Simulated Annealing with improved strategy
    def simulated_annealing(start_h, max_iters=2000, T_start=0.08, T_end=0.0005, cooling=0.9995, reheat_every=500, reheat_T=0.03):
        """Simulated annealing on the 600 heights with reheat mechanism."""
        h_sa = start_h.copy()
        current_score = compute_score(h_sa)
        best_h_sa = h_sa.copy()
        best_score_sa = current_score
        T = T_start
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Reheat periodically to escape local minima
            if it > 0 and it % reheat_every == 0:
                T = reheat_T
            
            # Adaptive perturbation size based on current temperature
            n_perturb = rng.integers(20, 150)
            indices = rng.choice(N, size=n_perturb, replace=False)
            
            # Scale noise with temperature and current height std
            noise_scale = 0.1 * T * (np.std(h_sa) + 0.01)
            h_new = h_sa.copy()
            h_new[indices] += rng.normal(0, noise_scale, size=n_perturb)
            h_new = np.maximum(h_new, 0.0)
            
            if np.sum(h_new) <= 1e-10:
                continue
            
            new_score = compute_score(h_new)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                h_sa = h_new
                current_score = new_score
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_h_sa = h_sa.copy()
            
            T *= cooling
        
        return best_h_sa, best_score_sa
    
    # Run simulated annealing from multiple diverse starts
    sa_starts = [best_h.copy()]
    
    # Add diverse starts with different structures
    for i in range(8):
        h_start = np.ones(N)
        x = np.linspace(0, 1, N)
        mode = rng.integers(1, 6)
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.2, 0.5)
        h_start += amp * np.cos(2 * np.pi * mode * x + phase)
        h_start = np.maximum(h_start, 0.0)
        if np.sum(h_start) > 0:
            sa_starts.append(h_start)
    
    for h_start in sa_starts:
        if time_left() < 25:
            break
        
        h_sa, score_sa = simulated_annealing(h_start, max_iters=1200, T_start=0.08, T_end=0.0005, cooling=0.9995, reheat_every=400, reheat_T=0.03)
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
    
    # Phase 3: Gradient-based optimization with adaptive learning rate and line search
    h_opt = best_h.copy()
    alpha = 50.0
    lr = 0.01
    
    for it in range(400):
        if time_left() < 10:
            break
        
        grad = _smooth_grad(h_opt, alpha, M)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Line search: try multiple step sizes
        best_lr_this_iter = lr
        best_score_this_iter = compute_score(h_opt)
        
        for trial_lr in [lr * 2.0, lr, lr * 0.5, lr * 0.25]:
            h_trial = h_opt - trial_lr * grad
            h_trial = np.maximum(h_trial, 0.0)
            
            score_trial = compute_score(h_trial)
            if score_trial < best_score_this_iter:
                best_score_this_iter = score_trial
                best_lr_this_iter = trial_lr
                h_opt = h_trial
        
        if best_score_this_iter < best_score:
            best_score = best_score_this_iter
            best_h = h_opt.copy()
        
        # Adapt learning rate
        if best_score_this_iter < compute_score(best_h + 1e-10 * rng.standard_normal(N)):
            lr = min(lr * 1.1, 0.05)
        else:
            lr = max(lr * 0.9, 0.001)
        
        if it % 20 == 0:
            alpha = min(alpha * 1.2, 500.0)
    
    # Phase 4: More gradient optimization with different alphas
    for alpha_try in [100.0, 200.0, 400.0]:
        if time_left() < 15:
            break
        
        h_try = best_h.copy()
        lr_try = 0.005
        
        for it in range(150):
            if time_left() < 5:
                break
            
            grad = _smooth_grad(h_try, alpha_try, M)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_try = h_try - lr_try * grad
            h_try = np.maximum(h_try, 0.0)
            
            score = compute_score(h_try)
            if score < best_score:
                best_score = score
                best_h = h_try.copy()
            
            lr_try = max(lr_try * 0.97, 0.0005)
    
    # Phase 5: Two-block optimization
    if time_left() > 15:
        for gap_frac in np.linspace(0.05, 0.4, 8):
            if time_left() < 10:
                break
            for block_frac in np.linspace(0.1, 0.4, 6):
                if time_left() < 8:
                    break
                block_len = int(block_frac * N)
                gap_len = int(gap_frac * N)
                if block_len < 2 or gap_len < 1:
                    continue
                total_len = 2 * block_len + gap_len
                if total_len > N:
                    continue
                start = (N - total_len) // 2
                
                h_two = np.zeros(N)
                h_two[start:start+block_len] = 1.0
                h_two[start+block_len+gap_len:start+2*block_len+gap_len] = 1.0
                
                score = compute_score(h_two)
                if score < best_score:
                    best_score = score
                    best_h = h_two.copy()
                
                # Tapered two-block
                if time_left() < 6:
                    continue
                h_taper = np.zeros(N)
                ramp_len = max(2, block_len // 4)
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, ramp_len)))
                
                pos1 = start
                pos2 = start + block_len + gap_len
                for offset in range(block_len):
                    if offset < ramp_len:
                        h_taper[pos1 + offset] = ramp[offset]
                        h_taper[pos2 + offset] = ramp[offset]
                    elif offset >= block_len - ramp_len:
                        h_taper[pos1 + offset] = ramp[block_len - 1 - offset]
                        h_taper[pos2 + offset] = ramp[block_len - 1 - offset]
                    else:
                        h_taper[pos1 + offset] = 1.0
                        h_taper[pos2 + offset] = 1.0
                
                score = compute_score(h_taper)
                if score < best_score:
                    best_score = score
                    best_h = h_taper.copy()
        
        # Optimize best two-block with gradient
        if time_left() > 10:
            h_two_opt = best_h.copy()
            alpha_t = 100.0
            lr_t = 0.005
            
            for it in range(100):
                if time_left() < 5:
                    break
                
                grad = _smooth_grad(h_two_opt, alpha_t, M)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-15:
                    break
                
                h_two_opt = h_two_opt - lr_t * grad
                h_two_opt = np.maximum(h_two_opt, 0.0)
                
                score = compute_score(h_two_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_two_opt.copy()
                
                lr_t = max(lr_t * 0.97, 0.0005)
    
    # Phase 6: Final high-precision polish with very fine steps and increasing alpha
    if time_left() > 10:
        h_final = best_h.copy()
        
        for alpha_f in [500.0, 1000.0, 2000.0]:
            if time_left() < 5:
                break
            
            lr_f = 0.0003
            
            for it in range(300):
                if time_left() < 3:
                    break
                
                grad = _smooth_grad(h_final, alpha_f, M)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-18:
                    break
                
                h_final = h_final - lr_f * grad
                h_final = np.maximum(h_final, 0.0)
                
                score = compute_score(h_final)
                if score < best_score:
                    best_score = score
                    best_h = h_final.copy()
                
                lr_f = max(lr_f * 0.995, 0.00005)
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
