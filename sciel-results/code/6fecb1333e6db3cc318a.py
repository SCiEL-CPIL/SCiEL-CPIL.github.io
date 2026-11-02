import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _fft_convolve(h):
    """Compute full autoconvolution via FFT, returning array of length 2N-1."""
    n = len(h)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, fft_size)
    conv = np.fft.irfft(H * np.conj(H), fft_size)
    return conv[:size]

def compute_score(h, exclude_lag0=True):
    """
    Compute the normalized max autoconvolution.
    If exclude_lag0 is True, max is over k >= 1.
    Normalization: (sum h)^2.
    """
    h = np.asarray(h, dtype=np.float64)
    S_total = np.sum(h)
    if S_total < 1e-9:
        return np.inf
    S = _fft_convolve(h)
    n = len(h)
    if exclude_lag0:
        # Exclude index n-1 (lag 0)
        if n == 1:
            return 0.0
        # Max over all indices except n-1
        # Due to symmetry, we can just take max of S[:n-1] and S[n:]
        # But let's just do it properly
        S_no0 = np.concatenate([S[:n-1], S[n:]])
        max_side = np.max(S_no0)
    else:
        max_side = np.max(S)
    return max_side / (S_total ** 2)

def _smooth_max(x, beta=50.0):
    """Smooth approximation of max using soft-max."""
    return np.logaddexp.reduce(np.log(x + 1e-15)) * beta

def _objective_and_grad(h, beta=50.0):
    """
    Objective: normalized max of autoconvolution (excluding lag 0) using soft-max.
    Returns (value, gradient).
    """
    h = np.maximum(h, 1e-9)  # Ensure positivity for log
    S_total = np.sum(h)
    if S_total < 1e-9:
        return 1e10, np.zeros_like(h)
    
    S = _fft_convolve(h)
    n = len(h)
    
    # Exclude lag 0 (index n-1)
    S_no0 = np.concatenate([S[:n-1], S[n:]])
    
    # Soft-max for smooth max
    beta_val = beta
    log_sum_exp = np.logaddexp.reduce(np.log(S_no0 + 1e-15))
    smooth_max_val = np.exp(log_sum_exp)
    
    # Gradient of soft-max: d/dh_i [exp(logsumexp(log(S+eps)))]
    # = sum_j [S_j/(S_j+eps)] * d(S_j)/d(h_i) / smooth_max_val
    # Actually, d(soft_max)/d(S_k) = S_k / smooth_max_val
    # And d(S_k)/d(h_i) = h_{k-i} + h_{k+i} (for valid indices)
    
    # This is complex. Let's use a simpler approach:
    # Just compute the value, and use finite differences or a simpler gradient.
    # For speed, let's just return the value and use Nelder-Mead or similar derivative-free method.
    # Or, use a simpler objective: the actual max, but that's non-differentiable.
    
    # Let's use a different approach: minimize the max using a different parameterization.
    # Actually, let's just use the raw max and use a derivative-free optimizer.
    
    value = np.max(S_no0) / (S_total ** 2)
    
    # For gradient, let's compute it properly but efficiently
    # d/dh_i [max(S_no0) / S_total^2]
    # This is tricky. Let's use the soft-max gradient.
    
    # p_k = S_k / smooth_max_val for k in no0 indices
    p = (S_no0 + 1e-15) / smooth_max_val
    p_sum = np.sum(p)
    p = p / p_sum  # Normalize
    
    # d(smooth_max)/d(h_i) = sum_k p_k * d(S_k)/d(h_i)
    # d(S_k)/d(h_i) for k in [0, 2n-2], i in [0, n-1]:
    # S_k = sum_j h_j * h_{j+k-n+1} (with appropriate indexing)
    # This is getting complex. Let's just use a derivative-free method.
    
    return value, None

def _objective_scalar(h, beta=50.0):
    """Scalar objective for derivative-free optimization."""
    return compute_score(h, exclude_lag0=True)

def _make_initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Two disjoint blocks with various widths and separations
    for W_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        W = int(N * W_frac)
        for D_frac in [0.2, 0.3, 0.4, 0.5]:
            D = int(N * D_frac)
            h = np.zeros(N)
            start1 = max(0, N//2 - D - W//2)
            start2 = min(N - W, N//2 + D - W//2)
            if start1 + W > start2:
                continue
            h[start1:start1+W] = 1.0
            h[start2:start2+W] = 1.0
            if np.sum(h) > 0:
                candidates.append(h)
    
    # 3. Cosine window
    x = np.linspace(-1, 1, N)
    h_cos = np.maximum(0, np.cos(np.pi * x / 2))
    candidates.append(h_cos)
    
    # 4. Random binary
    for _ in range(5):
        h_rand = np.random.randint(0, 2, N).astype(float)
        if np.sum(h_rand) > 10:
            candidates.append(h_rand)
    
    # 5. Random smooth
    for _ in range(5):
        h_smooth = np.abs(np.random.randn(N))
        # Smooth it
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_smooth, kernel, mode='same')
        h_smooth = h_smooth / np.sum(h_smooth)
        candidates.append(h_smooth)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Evaluate all initial candidates
    candidates = _make_initial_candidates()
    for h_cand in candidates:
        if time.time() > deadline - 5:
            break
        h_cand = np.asarray(h_cand, dtype=np.float64)
        if np.sum(h_cand) < 1e-6:
            continue
        s = compute_score(h_cand, exclude_lag0=True)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Two-stage optimization:
    # Stage 1: Use L-BFGS-B with a smooth objective (soft-max) to get a good starting point
    # Stage 2: Use coordinate descent or local search to refine
    
    # For L-BFGS-B, we need a differentiable objective.
    # Let's use the soft-max based objective.
    
    def _soft_objective(h):
        h = np.maximum(h, 1e-9)
        S_total = np.sum(h)
        if S_total < 1e-9:
            return 1e10
        S = _fft_convolve(h)
        n = len(h)
        S_no0 = np.concatenate([S[:n-1], S[n:]])
        # Soft max with beta
        beta = 100.0
        # To avoid overflow, use log-sum-exp
        S_safe = S_no0 + 1e-15
        log_S = np.log(S_safe)
        log_max = np.max(log_S)
        sum_exp = np.sum(np.exp(log_S - log_max))
        soft_max = log_max + np.log(sum_exp)
        return np.exp(soft_max) / (S_total ** 2)
    
    def _soft_objective_grad(h):
        h = np.maximum(h, 1e-9)
        S_total = np.sum(h)
        if S_total < 1e-9:
            return np.zeros_like(h)
        S = _fft_convolve(h)
        n = len(h)
        S_no0 = np.concatenate([S[:n-1], S[n:]])
        beta = 100.0
        
        S_safe = S_no0 + 1e-15
        log_S = np.log(S_safe)
        log_max = np.max(log_S)
        exp_S = np.exp(log_S - log_max)
        sum_exp = np.sum(exp_S)
        p = exp_S / sum_exp  # Weights for soft-max
        
        # d(soft_max)/d(h_i) = sum_k p_k * d(S_k)/d(h_i)
        # S_k for k in [0, 2n-2], but we excluded index n-1
        # S_k = sum_{j} h_j * h_{j + k - (n-1)} where indices are valid
        # This is complex. Let's use a simpler gradient approximation.
        
        # Actually, let's just compute the gradient numerically if needed,
        # or use a simpler approach.
        
        # For now, return a zero gradient and use a derivative-free method.
        return np.zeros_like(h)
    
    # Let's use Nelder-Mead instead, which is derivative-free
    for h_start in [best_h]:
        if time.time() > deadline - 30:
            break
        
        try:
            # Normalize
            h_norm = h_start / np.sum(h_start)
            
            res = minimize(_objective_scalar, h_norm, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or hasattr(res, 'x'):
                h_opt = np.maximum(res.x, 0)
                if np.sum(h_opt) > 1e-6:
                    s = compute_score(h_opt, exclude_lag0=True)
                    if s < best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local search: coordinate descent on continuous values
    if time.time() < deadline - 15:
        h_curr = best_h.copy()
        s_curr = best_score
        step_size = 0.05
        patience = 0
        
        for iteration in range(20000):
            if time.time() > deadline - 5:
                break
            
            # Random coordinate
            idx = np.random.randint(N)
            old_val = h_curr[idx]
            
            # Try a few perturbations
            improvements = 0
            for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                new_val = max(0, old_val + delta)
                h_curr[idx] = new_val
                s_new = compute_score(h_curr, exclude_lag0=True)
                
                if s_new < s_curr:
                    s_curr = s_new
                    if s_new < best_score:
                        best_score = s_new
                        best_h = h_curr.copy()
                    improvements += 1
                    break
                else:
                    h_curr[idx] = old_val
            
            if improvements == 0:
                patience += 1
                # Reduce step size occasionally
                if patience % 100 == 0:
                    step_size *= 0.9
            else:
                patience = 0
        
        # Ensure best_h is valid
        if np.sum(best_h) < 1e-6:
            best_h = np.ones(N) / N
    
    return list(best_h)
