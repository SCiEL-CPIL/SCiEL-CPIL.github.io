import numpy as np
import time
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _setup_fft():
    global _FFT_SIZE
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _autoconv_and_grad(h):
    """Compute autoconvolution and its gradient with respect to h."""
    conv = _fft_convolve(h)
    # Gradient of conv[k] = sum_i h[i]*h[k-i] w.r.t. h[j]
    # d conv[k] / d h[j] = h[k-j] + h[j-k] (for valid indices)
    # Using FFT: gradient of conv is 2 * ifft(fft(h) * fft(h)) ... actually
    # conv = h * h (linear convolution), so d(conv)/d(h[j]) = 2*h shifted
    # More precisely: conv[k] = sum_i h[i] h[k-i]
    # d conv[k] / d h[j] = h[k-j] + h[j-k] (with h out of range = 0)
    # This is equivalent to 2 * (h * h) evaluated appropriately
    # Actually the gradient of the full conv vector w.r.t h is a matrix.
    # For the max, we need gradient of max(conv) which is conv at argmax.
    # d max(conv) / d h[j] = sum_k delta(k, argmax) * d conv[k]/d h[j]
    # = d conv[argmax]/d h[j] = h[argmax-j] + h[j-argmax] (with bounds)
    return conv

def _two_block_optimize(h_init, deadline, max_iter=50):
    """
    Two-block optimization: alternate between optimizing the 'peak' region
    and the 'tail' region of the height function.
    
    The idea: the autoconvolution maximum is often dominated by a few peaks.
    We can optimize the function in blocks, focusing on reducing the max.
    """
    best_h = h_init.copy()
    best_score = compute_score(best_h)
    
    # Split into two blocks: first half and second half
    mid = N // 2
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Block 1: optimize first half, keep second half fixed
        def obj1(h1):
            h = np.concatenate([h1, best_h[mid:]])
            s = np.sum(h)
            if s < 1e-10:
                return 1.0
            conv = _fft_convolve(h)
            return np.max(conv) / (s * s)
        
        def grad1(h1):
            h = np.concatenate([h1, best_h[mid:]])
            s = np.sum(h)
            if s < 1e-10:
                return np.zeros(mid)
            conv = _fft_convolve(h)
            kmax = np.argmax(conv)
            # Gradient of conv[kmax] w.r.t h[0:mid]
            # d conv[kmax]/d h[j] = h[kmax-j] + h[j-kmax] for j in [0, mid)
            g = np.zeros(mid)
            for j in range(mid):
                idx1 = kmax - j
                idx2 = j - kmax
                if 0 <= idx1 < N:
                    g[j] += h[idx1]
                if 0 <= idx2 < N:
                    g[j] += h[idx2]
            return g / (s * s)
        
        try:
            res1 = optimize.minimize(
                obj1, best_h[:mid],
                jac=grad1,
                method='L-BFGS-B',
                bounds=[(0, 1.0)] * mid,
                options={'maxiter': 15, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h_new = np.concatenate([res1.x, best_h[mid:]])
            s = np.sum(h_new)
            if s > 1e-10:
                score_new = compute_score(h_new)
                if score_new < best_score:
                    best_h = h_new.copy()
                    best_score = score_new
        except Exception:
            pass
        
        # Block 2: optimize second half, keep first half fixed
        def obj2(h2):
            h = np.concatenate([best_h[:mid], h2])
            s = np.sum(h)
            if s < 1e-10:
                return 1.0
            conv = _fft_convolve(h)
            return np.max(conv) / (s * s)
        
        def grad2(h2):
            h = np.concatenate([best_h[:mid], h2])
            s = np.sum(h)
            if s < 1e-10:
                return np.zeros(N - mid)
            conv = _fft_convolve(h)
            kmax = np.argmax(conv)
            g = np.zeros(N - mid)
            for j in range(N - mid):
                j_global = mid + j
                idx1 = kmax - j_global
                idx2 = j_global - kmax
                if 0 <= idx1 < N:
                    g[j] += h[idx1]
                if 0 <= idx2 < N:
                    g[j] += h[idx2]
            return g / (s * s)
        
        try:
            res2 = optimize.minimize(
                obj2, best_h[mid:],
                jac=grad2,
                method='L-BFGS-B',
                bounds=[(0, 1.0)] * (N - mid),
                options={'maxiter': 15, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h_new = np.concatenate([best_h[:mid], res2.x])
            s = np.sum(h_new)
            if s > 1e-10:
                score_new = compute_score(h_new)
                if score_new < best_score:
                    best_h = h_new.copy()
                    best_score = score_new
        except Exception:
            pass
    
    return best_h, best_score

def _kkt_dual_ascent(h_init, deadline, max_iter=200):
    """
    KKT-based dual ascent for the problem:
    min_h max_k conv[k] / (sum h)^2
    subject to h >= 0
    
    We use a projected gradient method on the dual variables.
    The dual problem involves finding weights on the autoconvolution
    coefficients that minimize the dual objective.
    
    Simplified approach: iterate on h using a smoothed max objective
    with adaptive temperature, using projected gradient descent.
    """
    best_h = h_init.copy()
    best_score = compute_score(best_h)
    
    beta = 20.0  # smoothing parameter
    lr = 0.1
    patience = 30
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        s = np.sum(best_h)
        if s < 1e-10:
            break
        
        # Compute smooth max of conv
        conv = _fft_convolve(best_h)
        max_c = np.max(conv)
        exps = np.exp(beta * (conv - max_c))
        log_sum_exp = max_c + (1.0 / beta) * np.log(np.sum(exps))
        
        # Gradient of smooth max w.r.t. h
        # d log_sum_exp / d conv[k] = exps[k] / sum(exps)
        weights = exps / np.sum(exps)
        
        # d conv[k] / d h[j] = h[k-j] + h[j-k] (with bounds)
        grad = np.zeros(N)
        for j in range(N):
            for k in range(_CONV_LEN):
                if weights[k] < 1e-10:
                    continue
                idx1 = k - j
                idx2 = j - k
                gk = 0.0
                if 0 <= idx1 < N:
                    gk += best_h[idx1]
                if 0 <= idx2 < N:
                    gk += best_h[idx2]
                grad[j] += weights[k] * gk
        
        # Normalize by (s^2)
        grad /= (s * s)
        
        # Projected gradient step
        h_new = best_h - lr * grad
        h_new = np.maximum(h_new, 0.0)  # project onto h >= 0
        
        s_new = np.sum(h_new)
        if s_new < 1e-10:
            break
        
        score_new = compute_score(h_new)
        
        if score_new < best_score:
            best_h = h_new.copy()
            best_score = score_new
            no_improve = 0
        else:
            no_improve += 1
            # Reduce learning rate if no improvement
            if no_improve % 10 == 0:
                lr *= 0.8
                beta *= 1.5  # increase smoothing for harder optimization
        
        if no_improve > patience:
            break
    
    return best_h, best_score

def _bisection_lp(h_init, deadline):
    """
    Bisection on the target value T:
    Check if there exists h >= 0 such that conv[k] <= T * (sum h)^2 for all k.
    
    This is a feasibility problem. We use a simple iterative method:
    Start with h_init, and iteratively reduce components that cause
    conv[k] to exceed the threshold.
    """
    # First, find a good upper bound on the score
    s_init = compute_score(h_init)
    T_low = 0.0
    T_high = s_init
    
    # Binary search on T
    for _ in range(30):
        if time.time() > deadline:
            break
        
        T_mid = (T_low + T_high) / 2
        
        # Try to find h such that max conv[k] / (sum h)^2 <= T_mid
        # Use iterative water-filling style approach
        h = h_init.copy()
        s = np.sum(h)
        
        feasible = True
        for _ in range(50):
            if time.time() > deadline:
                return h_init, compute_score(h_init)
            
            conv = _fft_convolve(h)
            target = T_mid * (s * s)
            
            # Find the maximum violation
            violations = conv - target
            max_viol = np.max(violations)
            
            if max_viol <= 1e-8:
                feasible = True
                break
            
            # Reduce the contribution to the violating lags
            kmax = np.argmax(violations)
            
            # For the lag kmax, reduce h components that contribute most
            # d conv[kmax] / d h[j] = h[kmax-j] + h[j-kmax]
            contrib = np.zeros(N)
            for j in range(N):
                idx1 = kmax - j
                idx2 = j - kmax
                c = 0.0
                if 0 <= idx1 < N:
                    c += h[idx1]
                if 0 <= idx2 < N:
                    c += h[idx2]
                contrib[j] = c
            
            # Reduce h where contribution is highest
            # Use a soft approach: h[j] -= alpha * contrib[j] for large contrib
            alpha = max_viol / (np.sum(contrib) + 1e-10) * 0.5
            h = np.maximum(h - alpha * contrib, 0.0)
            
            s = np.sum(h)
            if s < 1e-10:
                feasible = False
                break
        
        if feasible:
            T_high = T_mid
        else:
            T_low = T_mid
    
    # Return the best h found
    return h_init, compute_score(h_init)

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    _setup_fft()
    
    # Initial candidates
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    candidates = []
    candidates.append(np.ones(N) / N)
    
    sigma_g = 0.1
    h_gauss = np.exp(-x**2 / (2 * sigma_g**2))
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    h_u = x**2
    h_u /= np.sum(h_u)
    candidates.append(h_u)
    
    h_sin = np.sin(np.pi * x / 0.25)**2
    h_sin /= np.sum(h_sin)
    candidates.append(h_sin)
    
    # Try a few more structured candidates
    h_tri = 1.0 - np.abs(x) / 0.25
    h_tri = np.maximum(h_tri, 0)
    h_tri /= np.sum(h_tri)
    candidates.append(h_tri)
    
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Two-block optimization
    if time.time() < deadline - 30:
        h2, s2 = _two_block_optimize(best_h, deadline - 10)
        if s2 < best_score:
            best_score = s2
            best_h = h2.copy()
    
    # KKT dual ascent
    if time.time() < deadline - 20:
        h3, s3 = _kkt_dual_ascent(best_h, deadline - 5)
        if s3 < best_score:
            best_score = s3
            best_h = h3.copy()
    
    # Bisection LP (if time remains)
    if time.time() < deadline - 10:
        h4, s4 = _bisection_lp(best_h, deadline - 2)
        if s4 < best_score:
            best_score = s4
            best_h = h4.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    return best_h.tolist()
