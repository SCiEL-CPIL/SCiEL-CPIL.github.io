"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Structural change: use a reduced-dimensionality spectral parameterization
(multi-sine with random phases) to find good starting points via global
optimization, then refine in full dimension with L-BFGS on a smooth
surrogate (soft-max of autoconvolution), then polish discretely.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _soft_max_abs(ac, t=20.0):
    """Smooth approximation to max|ac| using log-sum-exp."""
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-12:
        return 0.0
    return m + np.log(np.sum(np.exp((x - m) / t))) * t / len(x)


def _smooth_obj(f, t=10.0):
    """
    Smooth objective: soft_max|ac| / (sum f)^2
    Returns (value, gradient).
    """
    s = np.sum(f)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(f)
    
    ac = _autoconv_fft(f)
    x = np.abs(ac)
    m = np.max(x)
    
    if m < 1e-12:
        val = 0.0
        grad_ac = np.zeros_like(ac)
    else:
        # log-sum-exp for stability
        shifted = x / t - m / t
        exp_shifted = np.exp(shifted)
        log_sum = m / t + np.log(np.sum(exp_shifted))
        val = t * log_sum / (s * s)
        
        # Gradient w.r.t. ac: (1/s^2) * (1/t) * exp_shifted / sum(exp_shifted) * sign(ac)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_ac = (weights * np.sign(ac)) / (s * s)
    
    # Gradient of ac w.r.t. f: for linear autoconvolution ac = f * f
    # d(ac[k])/d(f[j]) = 2*f[k-j] for valid k-j (with the k=2j case giving 2*f[j])
    # This is equivalent to: grad_f[j] = 2 * sum_k grad_ac[k] * f[k-j] for valid indices
    # Which is the cross-correlation of grad_ac with f.
    
    # Compute via FFT: the gradient of sum_k g[k]*ac[k] w.r.t. f[j]
    # where ac = f*f (linear conv), is 2 * (g * f)[j] where * is appropriate correlation
    # Actually: d/d(f[j]) sum_k g[k] ac[k] = 2 * sum_k g[k] f[k-j] (for valid k-j)
    # = 2 * (g_rev * f)[j] where g_rev[k] = g[2N-2-k]... 
    # Let's just use: the j-th element of 2 * np.convolve(g, f, mode='full')[:N]
    # Wait, need to be careful. Let me use the standard result:
    # If ac[k] = sum_i f[i] f[k-i], then d/d(f[j]) sum_k g[k] ac[k] 
    #   = sum_k g[k] (f[k-j] + f[j-k]) for valid indices
    #   = 2 * sum_k g[k] f[k-j] (by symmetry of the sum, since g is defined on [0,2N-2])
    # The sum sum_k g[k] f[k-j] for k in [j, j+N-1] is the j-th element of the
    # convolution of g (padded) with f, or equivalently the cross-correlation.
    
    # Efficient computation:
    size = 1
    while size < 2 * N - 1:
        size *= 2
    G = np.fft.rfft(grad_ac, size)
    F = np.fft.rfft(f, size)
    # The result is 2 * ifft(G * conj(F))[:N] ... let me think again.
    # sum_k g[k] f[k-j]: let m = k-j, k = m+j, m in [0, N-1], k in [j, j+N-1]
    # = sum_m g[m+j] f[m]
    # This is the j-th element of the cross-correlation of g with f.
    # In FFT terms: ifft(conj(fft(g)) * fft(f)) gives the cross-correlation.
    
    corr = np.fft.irfft(np.conj(G) * F, size)
    grad_f = 2.0 * corr[:N]
    
    # Now the full gradient of val = soft_max / s^2:
    # d(val)/d(f[j]) = (grad_f[j] * s^2 - soft_max * 2*s) / s^4
    #                = grad_f[j] / s^2 - 2*soft_max / s^3
    
    soft_max_val = t * (m / t + np.log(np.sum(exp_shifted))) if m >= 1e-12 else 0.0
    grad = grad_f / (s * s) - 2.0 * soft_max_val / (s ** 3)
    
    return val, grad


def _spectral_signal(params, K, freqs, phases):
    """
    Generate a signal from spectral parameters.
    params: array of length 2*K (amplitudes and phase modulations)
    """
    amps = params[:K]
    phasemod = params[K:]
    
    f = np.zeros(N)
    for k in range(K):
        f += amps[k] * np.cos(2 * np.pi * freqs[k] * np.arange(N) / N + phases[k] + phasemod[k])
    return f


def _spectral_objective(params, K, freqs, phases):
    """Objective for spectral optimization."""
    f = _spectral_signal(params, K, freqs, phases)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _spectral_optimize(K, freqs, phases, max_iter=50):
    """
    Optimize spectral parameters using differential evolution.
    """
    if _remaining() < 15.0:
        return None, np.inf
    
    bounds = [(-1.0, 1.0)] * (2 * K)
    
    try:
        result = differential_evolution(
            _spectral_objective,
            bounds,
            args=(K, freqs, phases),
            maxiter=max_iter,
            tol=1e-4,
            seed=42,
            polish=False,
            workers=1
        )
        
        if result.success or result.fun < 1e10:
            f = _spectral_signal(result.x, K, freqs, phases)
            score = _score(np.sign(f))
            return np.sign(f), score
    except Exception:
        pass
    
    return None, np.inf


def _lbfgs_refine(f_init, max_iter=200):
    """
    Refine using L-BFGS-B on the smooth objective.
    """
    if _remaining() < 10.0:
        return f_init
    
    f = f_init.astype(float).copy()
    
    def obj_and_grad(x):
        val, grad = _smooth_obj(x, t=5.0)
        return val, grad
    
    try:
        result = minimize(
            obj_and_grad,
            f,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        return result.x
    except Exception:
        return f


def _discrete_polish(f, max_rounds=20):
    """
    Discrete local search with single-bit flips.
    """
    f = f.copy()
    s = np.sum(f)
    if abs(s) < 1e-12:
        return f, np.inf
    
    ac = _autoconv_fft(f)
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    for _ in range(max_rounds):
        if _remaining() < 5.0:
            break
        
        improved = False
        for idx in range(N):
            f[idx] = -f[idx]
            s_new = np.sum(f)
            if abs(s_new) < 1e-12:
                f[idx] = -f[idx]
                continue
            ac_new = _autoconv_fft(f)
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < cur_score:
                cur_score = new_score
                ac = ac_new
                s = s_new
                improved = True
            else:
                f[idx] = -f[idx]
        
        if not improved:
            break
    
    return f, cur_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    rng = np.random.RandomState(42)
    
    best_f = None
    best_score = np.inf
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Evaluate all candidates
    for c in candidates:
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_f = c.copy()
    
    # Spectral-domain optimization with multiple frequency sets
    # Use different numbers of sinusoidal components
    for K in [10, 14, 18]:
        if _remaining() < 25.0:
            break
        
        # Generate random frequency sets (avoiding DC and Nyquist)
        for trial in range(3):
            if _remaining() < 20.0:
                break
            
            # Random frequencies in (0, 0.5) avoiding 0 and 0.5
            freqs = np.sort(rng.uniform(0.01, 0.49, K))
            phases = rng.uniform(0, 2 * np.pi, K)
            
            f_spec, score_spec = _spectral_optimize(K, freqs, phases, max_iter=30)
            
            if f_spec is not None and score_spec < best_score:
                best_score = score_spec
                best_f = f_spec.copy()
    
    # L-BFGS refinement from the best discrete solution
    if _remaining() > 20.0:
        f_cont = best_f.astype(float)
        f_refined = _lbfgs_refine(f_cont, max_iter=300)
        f_proj = np.sign(f_refined)
        f_proj[f_proj == 0] = 1.0
        score_proj = _score(f_proj)
        if score_proj < best_score:
            best_score = score_proj
            best_f = f_proj.copy()
    
    # Also try L-BFGS from a few random continuous starts
    for r in range(3):
        if _remaining() < 15.0:
            break
        
        f_init = rng.randn(N) * 0.5
        f_refined = _lbfgs_refine(f_init, max_iter=200)
        f_proj = np.sign(f_refined)
        f_proj[f_proj == 0] = 1.0
        score_proj = _score(f_proj)
        if score_proj < best_score:
            best_score = score_proj
            best_f = f_proj.copy()
    
    # Final discrete polish from the best solution
    if _remaining() > 10.0:
        f, score_final = _discrete_polish(best_f, max_rounds=30)
        if score_final < best_score:
            best_score = score_final
            best_f = f.copy()
    
    return best_f
