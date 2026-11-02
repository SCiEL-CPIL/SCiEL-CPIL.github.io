import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 600
DEADLINE = 110.0  # Safety margin before 120s kill

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
    # At t=0: sum_i h_i^2 * w = n * 1 * w = 0.5
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

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    # Efficient gradient using FFT-based correlation
    # grad[i] = 2 * f[peak_idx - i] if 0 <= peak_idx - i < N, else 0
    # This is a flipped and shifted version of f
    n = len(heights)
    grad = np.zeros(n, dtype=np.float64)
    
    # For each i, j = peak_idx - i must be in [0, n-1]
    # So i must be in [peak_idx - n + 1, peak_idx] intersect [0, n-1]
    i_min = max(0, peak_idx - n + 1)
    i_max = min(n - 1, peak_idx)
    if i_min <= i_max:
        indices = np.arange(i_min, i_max + 1)
        j_indices = peak_idx - indices
        grad[indices] = 2.0 * heights[j_indices]
    
    return conv, grad, peak_idx

def _simulated_annealing_optimize(start_time, deadline):
    """Multi-start simulated annealing that perturbs step heights with Gaussian noise."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    def compute_score_fast(heights):
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        max_conv = np.max(conv)
        total = np.sum(heights)
        if total <= 0:
            return np.inf
        return max_conv * N / (total * total * w)
    
    best_score = np.inf
    best_h = None
    
    rng = np.random.default_rng(42)
    
    # Multiple starts from different initial configurations
    for start_idx in range(5):
        if time_left() < 30:
            break
        
        # Different initial configurations
        if start_idx == 0:
            h = np.ones(N, dtype=np.float64)
        elif start_idx == 1:
            x = np.linspace(0, 1, N)
            h = np.exp(-((x - 0.5) / 0.3) ** 2)
        elif start_idx == 2:
            h = np.zeros(N, dtype=np.float64)
            h[:N//3] = 1.0
            h[2*N//3:] = 1.0
        elif start_idx == 3:
            x = np.linspace(0, 1, N)
            h = 1.0 - 0.5 * (2 * x - 1) ** 2
            h = np.maximum(h, 0.0)
        else:
            h = np.ones(N, dtype=np.float64)
            h[:N//4] = 0.5
            h[3*N//4:] = 0.5
        
        h = np.maximum(h, 0.0)
        if np.sum(h) <= 0:
            continue
        
        current_score = compute_score_fast(h)
        current_h = h.copy()
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
        
        # Simulated annealing parameters
        T_start = 0.01
        T_end = 0.0001
        cooling_rate = 0.995
        n_iterations = 2000
        
        T = T_start
        
        for it in range(n_iterations):
            if time_left() < 15:
                break
            
            # Generate perturbation
            noise_scale = 0.05 * T / T_start
            h_new = current_h + rng.normal(0, noise_scale, size=N)
            h_new = np.maximum(h_new, 0.0)
            
            new_score = compute_score_fast(h_new)
            
            # Accept or reject
            if new_score < current_score:
                # Always accept improvement
                current_h = h_new
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                # Accept with probability exp(-(new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    accept_prob = np.exp(-delta / T)
                    accept = rng.random() < accept_prob
                
                if accept:
                    current_h = h_new
                    current_score = new_score
            
            # Cool down
            T = max(T * cooling_rate, T_end)
    
    return best_score, best_h

def _bisection_lp_optimize(start_time, deadline):
    """Bisection on T with LP to find minimum T such that exists h >= 0 with autoconv <= T."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    def check_feasible(T):
        """Check if there exists h >= 0 with sum(h) = 1 and max autoconv <= T."""
        # Build the autoconvolution matrix
        # conv_k = sum_i h_i * h_{k-i} for k in [0, 2N-2]
        # We need: for all k, sum_i h_i * h_{k-i} <= T
        # This is a quadratic constraint, so LP won't directly work
        # Instead, use a linear relaxation or iterative approach
        
        # For now, return False as a placeholder
        # We'll use a different approach below
        return False, None
    
    # Alternative: use the fact that for the optimal solution, 
    # the max autoconvolution is achieved at a few key points
    # and use iterative LP relaxation
    
    best_score = np.inf
    best_h = None
    
    # Try a simpler approach: optimize over a low-dimensional parameterization
    # Assume the optimal shape is piecewise linear with few pieces
    
    # Piecewise linear with k segments
    for k in [2, 3, 4, 5, 6, 8, 10]:
        if time_left() < 10:
            break
        
        # Create k-1 breakpoints in [0, 1]
        breakpoints = np.linspace(0, 1, k + 1)
        
        # Initial heights: try symmetric unimodal
        h_init = np.zeros(N)
        for i in range(k):
            seg_start = int(breakpoints[i] * N)
            seg_end = int(breakpoints[i + 1] * N)
            # Linear interpolation between segment heights
            h_start = np.sin(np.pi * (i + 1) / (k + 1))
            h_end = np.sin(np.pi * (i + 2) / (k + 1))
            for j in range(seg_start, min(seg_end, N)):
                t = (j - seg_start) / max(1, seg_end - seg_start)
                h_init[j] = h_start * (1 - t) + h_end * t
        
        h_init = np.maximum(h_init, 0)
        if np.sum(h_init) > 0:
            h_init = h_init / np.sum(h_init) * N  # normalize so sum = N for score=1 reference
        
        s = compute_score(h_init)
        if s < best_score:
            best_score = s
            best_h = h_init.copy()
    
    return best_score, best_h

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
    
    # Phase 1: Simulated annealing optimization
    if time_left() > 30:
        s_sa, h_sa = _simulated_annealing_optimize(start_time, deadline)
        if h_sa is not None and s_sa < best_score:
            best_score = s_sa
            best_h = h_sa.copy()
    
    # Phase 2: Structured shapes
    if time_left() > 15:
        rng = np.random.default_rng(12345)
        x = np.linspace(0, 1, N)
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
        
        # Gaussian bump
        for width in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            h_gauss = np.exp(-((x - 0.5) / width) ** 2)
            structured_candidates.append(h_gauss)
        
        # Parabolic
        for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_par = 1.0 - a * (2 * x - 1) ** 2
            h_par = np.maximum(h_par, 0.0)
            if np.max(h_par) > 0:
                structured_candidates.append(h_par)
        
        # Linear taper
        for slope in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_lin = 1.0 - slope * np.abs(2 * x - 1)
            h_lin = np.maximum(h_lin, 0.0)
            if np.max(h_lin) > 0:
                structured_candidates.append(h_lin)
        
        # Plateau with smooth edges
        for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h_plat = np.ones(N)
            edge_len = int((1 - plateau_frac) * N / 2)
            if edge_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
                h_plat[:edge_len] = ramp
                h_plat[-edge_len:] = ramp[::-1]
            structured_candidates.append(h_plat)
        
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
    
    # Phase 3: Gradient-based optimization with smooth surrogate
    if time_left() > 20:
        def smooth_max(conv, alpha):
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha * (conv - m))
            return m + np.log(np.sum(exp_conv)) / alpha
        
        def smooth_obj_and_grad(heights, alpha):
            f = np.zeros(M, dtype=np.float64)
            f[:N] = heights
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha * (conv - m))
            weights = exp_conv / np.sum(exp_conv)
            
            sm = smooth_max(conv, alpha)
            
            # Gradient of smooth_max w.r.t. f
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = np.sum(heights)
            w = 0.5 / N
            if S <= 1e-10:
                return 1e10, np.zeros(N)
            
            obj = sm / (S * w) ** 2 * (N * w)
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            return obj, obj_grad
        
        h_opt = best_h.copy()
        alpha = 50.0
        lr = 0.01
        
        for it in range(300):
            if time_left() < 10:
                break
            
            obj, grad = smooth_obj_and_grad(h_opt, alpha)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_opt = h_opt - lr * grad
            h_opt = np.maximum(h_opt, 0.0)
            
            score = compute_score(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
            
            if it % 20 == 0:
                alpha = min(alpha * 1.2, 500.0)
                lr = max(lr * 0.9, 0.001)
    
    # Phase 4: Random restarts
    if time_left() > 15:
        rng = np.random.default_rng(12345)
        
        for restart in range(10):
            if time_left() < 10:
                break
            
            freqs = np.arange(N)
            h_init = np.ones(N)
            num_modes = rng.integers(3, 8)
            for _ in range(num_modes):
                k = rng.integers(1, N // 4)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.05, 0.25)
                h_init += amp * np.cos(2 * np.pi * k * freqs / N + phase)
            h_init = np.maximum(h_init, 0.0)
            if np.sum(h_init) <= 0:
                continue
            
            h_r = h_init.copy()
            alpha_r = 50.0
            lr_r = 0.01
            
            for it in range(100):
                if time_left() < 10:
                    break
                
                f = np.zeros(M, dtype=np.float64)
                f[:N] = h_r
                F = np.fft.rfft(f)
                conv = np.fft.irfft(F * F, n=M)
                
                m = np.max(conv)
                if m < 0:
                    m = 0
                exp_conv = np.exp(alpha_r * (conv - m))
                weights = exp_conv / np.sum(exp_conv)
                sm = m + np.log(np.sum(exp_conv)) / alpha_r
                
                W_full = np.fft.rfft(weights)
                F_full = np.fft.rfft(f)
                corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
                grad_f = 2.0 * corr[:N]
                
                S = np.sum(h_r)
                w = 0.5 / N
                if S <= 1e-10:
                    break
                
                obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
                
                h_r = h_r - lr_r * obj_grad
                h_r = np.maximum(h_r, 0.0)
                
                score = compute_score(h_r)
                if score < best_score:
                    best_score = score
                    best_h = h_r.copy()
                
                if it % 20 == 0:
                    alpha_r = min(alpha_r * 1.15, 500.0)
                    lr_r = max(lr_r * 0.92, 0.001)
    
    # Final polish
    if time_left() > 10:
        h_final = best_h.copy()
        alpha_f = 200.0
        lr_f = 0.001
        
        for it in range(50):
            if time_left() < 5:
                break
            
            f = np.zeros(M, dtype=np.float64)
            f[:N] = h_final
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha_f * (conv - m))
            weights = exp_conv / np.sum(exp_conv)
            sm = m + np.log(np.sum(exp_conv)) / alpha_f
            
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = np.sum(h_final)
            w = 0.5 / N
            if S <= 1e-10:
                break
            
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            h_final = h_final - lr_f * obj_grad
            h_final = np.maximum(h_final, 0.0)
            
            score = compute_score(h_final)
            if score < best_score:
                best_score = score
                best_h = h_final.copy()
            
            lr_f = max(lr_f * 0.98, 0.0001)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
