import time
import numpy as np
from scipy.optimize import minimize

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
    
    # Phase 2: KKT dual optimization
    # The problem: min_h max_k (autoconv_k) subject to h >= 0
    # By duality, we can solve: max_lambda sum_k lambda_k * c_k - (1/2)||lambda||^2 subject to lambda >= 0
    # where c_k are related to the autoconvolution structure
    # The primal solution is h_i = max(0, sum_k lambda_k * A_{i,k}) where A encodes the correlation structure
    
    def _kkt_dual_optimize(start_h, max_iters=500, lr=0.01):
        """Optimize using KKT dual conditions with projected gradient ascent."""
        n = N
        w = 0.5 / n
        
        # Precompute the autoconvolution matrix structure
        # For step function with heights h, autoconv at lag k is:
        # (f*f)_k = w * sum_i h_i * h_{i+k} (for k >= 0)
        # The normalized score is max_k (f*f)_k / (sum(h)*w)^2 * n*w
        # = max_k (w * sum_i h_i * h_{i+k}) / (sum(h)*w)^2 * n*w
        # = max_k (sum_i h_i * h_{i+k}) / (sum(h))^2 * n
        
        # For the dual: we want to find weights lambda_k >= 0 on lags
        # such that the optimal h satisfies h_i = max(0, sum_k lambda_k * g_{i,k})
        # where g_{i,k} = h_{i+k} (but this is circular)
        
        # Alternative approach: Use the fact that at optimum, the gradient of the 
        # smooth max objective should be balanced with the nonnegativity constraint.
        # Use accelerated projected gradient on the primal with smooth surrogate.
        
        h_opt = start_h.copy()
        alpha = 100.0
        current_lr = lr
        best_h_local = h_opt.copy()
        best_score_local = compute_score(h_opt)
        
        # Momentum for accelerated gradient
        h_momentum = h_opt.copy()
        momentum = 0.9
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Compute gradient of smooth objective
            grad = _smooth_grad(h_momentum, alpha, M)
            
            # Update momentum
            h_momentum = h_momentum + momentum * (h_opt - h_momentum)
            h_momentum = h_momentum - current_lr * grad
            
            # Project onto nonnegative orthant
            h_opt = np.maximum(h_momentum, 0.0)
            
            # Ensure normalization helps: scale to maintain reasonable magnitude
            total = np.sum(h_opt)
            if total > 0:
                # Don't change the shape, just track
                pass
            
            score = compute_score(h_opt)
            if score < best_score_local:
                best_score_local = score
                best_h_local = h_opt.copy()
            
            # Adapt learning rate
            if it % 50 == 0:
                current_lr *= 0.8
                alpha = min(alpha * 1.3, 1000.0)
                momentum = min(momentum * 1.01, 0.95)
        
        return best_h_local, best_score_local
    
    # Run KKT dual optimization from the best structured candidate
    h_kkt, score_kkt = _kkt_dual_optimize(best_h.copy(), max_iters=400, lr=0.005)
    if score_kkt < best_score:
        best_score = score_kkt
        best_h = h_kkt.copy()
    
    # Also try from a few diverse starts
    for i in range(3):
        if time_left() < 25:
            break
        x = np.linspace(0, 1, N)
        h_diverse = np.ones(N)
        mode = rng.integers(1, 4)
        h_diverse += 0.2 * np.cos(2 * np.pi * mode * x)
        h_diverse = np.maximum(h_diverse, 0.0)
        
        h_kkt2, score_kkt2 = _kkt_dual_optimize(h_diverse, max_iters=300, lr=0.003)
        if score_kkt2 < best_score:
            best_score = score_kkt2
            best_h = h_kkt2.copy()
    
    # Phase 3: Refine with L-BFGS-B using smooth surrogate
    # This uses scipy's optimizer which handles the nonnegativity constraint properly
    
    def _smooth_obj_wrapper(h_flat):
        """Objective for scipy optimizer: smooth max of autoconvolution, normalized."""
        h = np.maximum(h_flat, 0.0)
        n = len(h)
        w = 0.5 / n
        f = np.zeros(M, dtype=np.float64)
        f[:n] = h
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Smooth max
        m = np.max(conv)
        if m < 0:
            m = 0
        alpha_lbfgs = 200.0
        exp_conv = np.exp(alpha_lbfgs * (conv - m))
        sm = m + np.log(np.sum(exp_conv)) / alpha_lbfgs
        
        total = np.sum(h)
        if total <= 1e-10:
            return 1e10
        
        val = sm / (total * w) ** 2 * (n * w)
        return val
    
    def _smooth_obj_grad_wrapper(h_flat):
        """Gradient for scipy optimizer."""
        h = np.maximum(h_flat, 0.0)
        n = len(h)
        w = 0.5 / n
        f = np.zeros(M, dtype=np.float64)
        f[:n] = h
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        alpha_lbfgs = 200.0
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha_lbfgs * (conv - m))
        weights = exp_conv / np.sum(exp_conv)
        sm = m + np.log(np.sum(exp_conv)) / alpha_lbfgs
        
        # Gradient of smooth max
        W_full = np.fft.rfft(weights)
        F_full = np.fft.rfft(f)
        corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
        grad_f = 2.0 * corr[:n]
        
        S = np.sum(h)
        if S <= 1e-10:
            return np.zeros(n)
        
        # d(sm)/(d h_i) * n/(S^2*w) - sm * 2*n/(S^3*w)
        obj_grad = grad_f * (n / (S * S * w)) - sm * (2.0 * n / (S * S * S * w))
        
        # Zero out gradient where h is at boundary (h=0)
        obj_grad[h < 1e-10] = 0.0
        return obj_grad
    
    if time_left() > 40:
        h_start = best_h.copy()
        bounds = [(0.0, None)] * N
        
        try:
            result = minimize(
                _smooth_obj_wrapper,
                h_start,
                jac=_smooth_obj_grad_wrapper,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
            )
            
            if result.success or result.fun < best_score:
                h_lbfgs = np.maximum(result.x, 0.0)
                score_lbfgs = compute_score(h_lbfgs)
                if score_lbfgs < best_score:
                    best_score = score_lbfgs
                    best_h = h_lbfgs.copy()
        except Exception:
            pass
    
    # Phase 4: Fine-tune with multiple alpha values
    for alpha_try in [300.0, 500.0, 800.0]:
        if time_left() < 15:
            break
        
        h_try = best_h.copy()
        lr_try = 0.003
        
        for it in range(80):
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
            
            lr_try = max(lr_try * 0.97, 0.0003)
    
    # Phase 5: Two-block optimization (keep as diversity source)
    if time_left() > 15:
        for gap_frac in np.linspace(0.05, 0.35, 6):
            if time_left() < 10:
                break
            for block_frac in np.linspace(0.15, 0.35, 5):
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
        
        # Optimize with gradient from two-block structure
        if time_left() > 12:
            h_two_opt = best_h.copy()
            alpha_t = 200.0
            lr_t = 0.003
            
            for it in range(60):
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
                
                lr_t = max(lr_t * 0.97, 0.0003)
    
    # Phase 6: Final high-precision polish
    if time_left() > 15:
        h_final = best_h.copy()
        alpha_f = 800.0
        lr_f = 0.0003
        
        for it in range(150):
            if time_left() < 5:
                break
            
            grad = _smooth_grad(h_final, alpha_f, M)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-16:
                break
            
            h_final = h_final - lr_f * grad
            h_final = np.maximum(h_final, 0.0)
            
            score = compute_score(h_final)
            if score < best_score:
                best_score = score
                best_h = h_final.copy()
            
            lr_f = max(lr_f * 0.99, 0.00005)
    
    # Phase 7: One more L-BFGS pass from the polished result
    if time_left() > 12:
        h_start2 = best_h.copy()
        bounds = [(0.0, None)] * N
        try:
            result2 = minimize(
                _smooth_obj_wrapper,
                h_start2,
                jac=_smooth_obj_grad_wrapper,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100}
            )
            if result2.success or result2.fun < best_score:
                h_final2 = np.maximum(result2.x, 0.0)
                score_final2 = compute_score(h_final2)
                if score_final2 < best_score:
                    best_score = score_final2
                    best_h = h_final2.copy()
        except Exception:
            pass
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
