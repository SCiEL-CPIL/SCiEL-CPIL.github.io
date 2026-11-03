import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _project_to_feasible(h):
    """
    Project h onto the feasible set: sum(h) = N/2, 0 <= h_i <= 1.
    Uses bisection on the threshold tau.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)
    
    # Correct any numerical drift in the sum
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-12:
        mask = (h_proj > 1e-8) & (h_proj < 1 - 1e-8)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _soft_max_obj_grad(h, beta):
    """
    Compute the log-sum-exp objective and its gradient.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _chirp_h(alpha, phase=0.0):
    """
    Generate a chirp signal: h_i = 0.5 + 0.5 * cos(2*pi*alpha*i^2/N + phase)
    This is a structural representation that inherently has low autocorrelation.
    """
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
    return h


def _optimize_chirp(alpha0, phase0=0.0, time_limit=10.0):
    """
    Optimize the chirp parameters (alpha, phase) to minimize the max overlap.
    Uses scipy.optimize.minimize with the log-sum-exp objective.
    """
    t_start = time.time()
    
    def obj(params):
        alpha, phase = params
        h = _chirp_h(alpha, phase)
        h = _project_to_feasible(h)
        val, _ = _soft_max_obj_grad(h, beta=100.0)
        return val
    
    def grad(params):
        alpha, phase = params
        h = _chirp_h(alpha, phase)
        h = _project_to_feasible(h)
        _, grad_h = _soft_max_obj_grad(h, beta=100.0)
        
        # Numerical gradient for alpha and phase
        eps = 1e-6
        h_a_plus = _project_to_feasible(_chirp_h(alpha + eps, phase))
        h_a_minus = _project_to_feasible(_chirp_h(alpha - eps, phase))
        h_p_plus = _project_to_feasible(_chirp_h(alpha, phase + eps))
        h_p_minus = _project_to_feasible(_chirp_h(alpha, phase - eps))
        
        v_a_plus, _ = _soft_max_obj_grad(h_a_plus, beta=100.0)
        v_a_minus, _ = _soft_max_obj_grad(h_a_minus, beta=100.0)
        v_p_plus, _ = _soft_max_obj_grad(h_p_plus, beta=100.0)
        v_p_minus, _ = _soft_max_obj_grad(h_p_minus, beta=100.0)
        
        grad_alpha = (v_a_plus - v_a_minus) / (2 * eps)
        grad_phase = (v_p_plus - v_p_minus) / (2 * eps)
        
        return np.array([grad_alpha, grad_phase])
    
    x0 = np.array([alpha0, phase0])
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                          options={'maxiter': 100, 'maxfun': 500, 'maxtime': time_limit})
        
        if result.success or result.fun < obj(x0):
            alpha_opt, phase_opt = result.x
            h_opt = _chirp_h(alpha_opt, phase_opt)
            h_opt = _project_to_feasible(h_opt)
            return h_opt, _score(h_opt)
    except Exception:
        pass
    
    # Fallback: return the initial chirp
    h_init = _chirp_h(alpha0, phase0)
    h_init = _project_to_feasible(h_init)
    return h_init, _score(h_init)


def _pgd_optimize(h0, beta, n_iter=500, lr=None, time_limit=30.0):
    """
    Projected Gradient Descent with adaptive learning rate.
    Returns the best h found.
    """
    h = _project_to_feasible(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    if lr is None:
        lr = 0.5 / (beta * N)
    
    t_start = time.time()
    
    for i in range(n_iter):
        if time.time() - t_start > time_limit:
            break
        
        val, grad = _soft_max_obj_grad(h, beta)
        
        # Update
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        # Check improvement
        new_val = _score(h_new)
        if new_val < best_val - 1e-12:
            best_val = new_val
            best_h = h_new.copy()
            h = h_new
            lr *= 1.01
        else:
            lr *= 0.8
            h = h_new
        
        if i % 50 == 0 and i > 0:
            recent_improvement = (best_val - val) > 1e-8
            if not recent_improvement and lr < 0.1:
                lr *= 1.5
    
    return best_h, best_val


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_to_feasible(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Chirp-based structural optimization (NEW APPROACH)
    # The chirp representation h_i = 0.5 + 0.5*cos(2*pi*alpha*i^2/N + phase)
    # has inherently low autocorrelation properties, which helps minimize overlap.
    
    if time.time() - t0 < _DEADLINE - 60:
        # Try multiple chirp initializations
        for alpha_init in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            for phase_init in [0.0, np.pi/2, np.pi]:
                if time.time() - t0 > _DEADLINE - 40:
                    break
                h_chirp, s_chirp = _optimize_chirp(alpha_init, phase_init, time_limit=5.0)
                _update_best(h_chirp)
    
    # 2. Analytical starts (fallback and diversification)
    i = np.arange(N)
    
    # Cosine wave (fundamental)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Higher harmonic
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)
    
    # Triangular wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * (i % N) / N - 1.0)
    _update_best(h_tri)
    
    # Random starts
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        _update_best(h_rand)
    
    # 3. PGD refinement from best candidate
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_refined, val_refined = _pgd_optimize(best_h, beta=50.0, n_iter=400, time_limit=15.0)
        _update_best(h_refined)
    
    # 4. Fine PGD with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_fine, val_fine = _pgd_optimize(best_h, beta=100.0, n_iter=500, time_limit=12.0)
        _update_best(h_fine)
    
    # 5. Very fine PGD with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        h_veryfine, val_veryfine = _pgd_optimize(best_h, beta=200.0, n_iter=600, time_limit=10.0)
        _update_best(h_veryfine)
    
    # 6. Final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        h_polish, val_polish = _pgd_optimize(best_h, beta=500.0, n_iter=800, time_limit=8.0)
        _update_best(h_polish)
    
    # 7. Structured perturbations with PGD refinement
    if time.time() - t0 < _DEADLINE - 10:
        for scale in [0.03, 0.015, 0.008]:
            if time.time() - t0 > _DEADLINE - 5:
                break
            for _ in range(3):
                if time.time() - t0 > _DEADLINE - 3:
                    break
                noise = np.random.normal(0, scale, N)
                h_pert = best_h + noise
                _update_best(h_pert)
                if time.time() - t0 < _DEADLINE - 3:
                    h_ref, val_ref = _pgd_optimize(h_pert, beta=100.0, n_iter=200, time_limit=5.0)
                    _update_best(h_ref)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project_to_feasible(best_h)
    return best_h
