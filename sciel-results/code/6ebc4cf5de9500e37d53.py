import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

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
    
    # Normalization: score = max_conv * n / (total^2 * w)
    score = max_conv * n / (total * total * w)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Precompute FFT size
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # The objective is: minimize max_t (f*f)(t) / (integral(f))^2 * support_length
    # Since support_length is fixed (0.5) and we can scale f freely,
    # we can fix the integral and minimize max autoconvolution.
    # Specifically: score = max_conv / (sum(h)*w)^2 * (N*w)
    # = max_conv * N / (sum(h)^2 * w)
    # If we fix sum(h) = S, then score = max_conv * N / (S^2 * w)
    # To minimize score, we want to minimize max_conv for a given S.
    # Since we can scale h, the ratio is invariant to scaling.
    # So we can fix sum(h) = 1 (or any constant) and minimize max_conv.
    
    # Strategy: 
    # 1. Start with good initial guesses (structured shapes)
    # 2. Use L-BFGS-B with finite differences to minimize max autoconv
    #    with constraint sum(h) = 1 and h >= 0
    # 3. The max is non-smooth, so use a smooth surrogate (log-sum-exp)
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N) / N  # uniform, sum = 1
    best_score = compute_score(best_h)
    
    def objective_and_grad(h, alpha=100.0):
        """Compute smooth max autoconvolution and its gradient.
        h is assumed to have sum=1 (handled externally or via penalty)."""
        f = np.zeros(M, dtype=np.float64)
        f[:N] = h
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Smooth max via log-sum-exp
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        sm = m + np.log(np.sum(exp_conv)) / alpha
        
        # Gradient of smooth max w.r.t. h
        weights = exp_conv / np.sum(exp_conv)
        
        # d(conv_k)/d(h_i) = 2*h[k-i] if 0 <= k-i < N
        # grad_h[i] = sum_k weights[k] * 2*h[k-i] (if valid)
        # = 2 * circular_correlation(weights, h)[i]
        W = np.fft.rfft(weights)
        H = np.fft.rfft(h)
        corr = np.fft.irfft(W * np.conj(H), n=M)
        grad = 2.0 * corr[:N]
        
        return sm, grad
    
    def make_obj(alpha):
        """Create objective function for L-BFGS-B."""
        def obj(h):
            sm, _ = objective_and_grad(h, alpha)
            return sm
        def obj_grad(h):
            _, g = objective_and_grad(h, alpha)
            return g
        return obj, obj_grad
    
    # Generate diverse initial guesses
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Cosine taper (smooth edges)
    x = np.linspace(0, 1, N)
    for taper_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h = np.ones(N)
        taper_len = int(taper_frac * N / 2)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h[:taper_len] = ramp
            h[-taper_len:] = ramp[::-1]
        h /= np.sum(h)
        candidates.append(h.copy())
    
    # 3. Gaussian-like
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        h = np.exp(-((x - 0.5) / sigma) ** 2)
        h /= np.sum(h)
        candidates.append(h.copy())
    
    # 4. Parabolic
    for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
        h = 1.0 - a * (2 * x - 1) ** 2
        h = np.maximum(h, 0.0)
        if np.sum(h) > 0:
            h /= np.sum(h)
            candidates.append(h.copy())
    
    # 5. Plateau with smooth edges
    for plateau_frac in [0.3, 0.5, 0.7, 0.9]:
        h = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h[:edge_len] = ramp
            h[-edge_len:] = ramp[::-1]
        h /= np.sum(h)
        candidates.append(h.copy())
    
    # 6. Random smooth functions
    for _ in range(15):
        h = np.ones(N)
        num_modes = rng.integers(2, 8)
        for _ in range(num_modes):
            k = rng.integers(1, N // 4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.3)
            h += amp * np.cos(2 * np.pi * k * x + phase)
        h = np.maximum(h, 0.0)
        if np.sum(h) > 0:
            h /= np.sum(h)
            candidates.append(h.copy())
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time_left() < 10:
            break
        score = compute_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Now optimize from the best candidate using L-BFGS-B
    # We minimize smooth max autoconv with constraint sum(h)=1, h>=0
    # Use penalty for sum constraint or reparameterize
    
    # Approach: minimize smooth_max_autoconv(h) subject to sum(h)=1, h>=0
    # L-BFGS-B supports bounds but not equality constraints directly.
    # Use a penalty term or fix one variable.
    
    # Better: use the fact that the problem is scale-invariant.
    # Fix sum(h) = 1 via normalization after each step, or use a Lagrange multiplier.
    
    # Let's use a simpler approach: minimize smooth_max_autoconv(h) + lambda*(sum(h)-1)^2
    # with h >= 0, using L-BFGS-B
    
    def penalized_obj(h, alpha, lam=100.0):
        sm, grad_sm = objective_and_grad(h, alpha)
        s = np.sum(h)
        penalty = lam * (s - 1.0) ** 2
        grad_penalty = 2.0 * lam * (s - 1.0) * np.ones(N)
        return sm + penalty, grad_sm + grad_penalty
    
    # Try multiple alphas and starting points
    for alpha in [50.0, 100.0, 200.0]:
        if time_left() < 15:
            break
        
        h_start = best_h.copy()
        
        # Add a small random perturbation to avoid exact local minima
        h_start = h_start + rng.normal(0, 0.001, N)
        h_start = np.maximum(h_start, 0.0)
        h_start /= np.sum(h_start)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                lambda h: penalized_obj(h, alpha)[0],
                h_start,
                jac=lambda h: penalized_obj(h, alpha)[1],
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_score:
                h_res = np.maximum(res.x, 0.0)
                h_res /= np.sum(h_res)
                score = compute_score(h_res)
                if score < best_score:
                    best_score = score
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Try from multiple good candidates
    if time_left() > 30:
        # Sort candidates by score and try top ones
        scored_candidates = []
        for h_cand in candidates:
            score = compute_score(h_cand)
            scored_candidates.append((score, h_cand))
        scored_candidates.sort(key=lambda x: x[0])
        
        for score_c, h_cand in scored_candidates[:5]:
            if time_left() < 15:
                break
            
            alpha = 100.0
            h_start = h_cand + rng.normal(0, 0.001, N)
            h_start = np.maximum(h_start, 0.0)
            h_start /= np.sum(h_start)
            
            try:
                res = minimize(
                    lambda h: penalized_obj(h, alpha)[0],
                    h_start,
                    jac=lambda h: penalized_obj(h, alpha)[1],
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.success or res.fun < best_score:
                    h_res = np.maximum(res.x, 0.0)
                    h_res /= np.sum(h_res)
                    score = compute_score(h_res)
                    if score < best_score:
                        best_score = score
                        best_h = h_res.copy()
            except Exception:
                pass
    
    # Final polish with higher alpha
    if time_left() > 10:
        alpha_final = 500.0
        h_start = best_h.copy()
        
        try:
            res = minimize(
                lambda h: penalized_obj(h, alpha_final)[0],
                h_start,
                jac=lambda h: penalized_obj(h, alpha_final)[1],
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < best_score:
                h_res = np.maximum(res.x, 0.0)
                h_res /= np.sum(h_res)
                score = compute_score(h_res)
                if score < best_score:
                    best_score = score
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Ensure non-negativity and normalize
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) > 0:
        best_h /= np.sum(best_h)
    
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
