"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov accelerated gradient descent on a smooth-max surrogate
with adaptive temperature, backtracking line search, and projection
onto the constraint set. Includes coordinate descent polish.
Added: spectral parameterization to reduce search space for faster
global optimization, followed by full-space polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
    
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)
    
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad = -(2.0 / N) * (A + B)
    
    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    This is equivalent to smooth max but with explicit t variable.
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp
    
    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)
    
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)
    
    return val, grad_h, grad_t


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.01):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    h_prev = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        step_lr = lr
        h_new = None
        for bt in range(20):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            h_new = y - lr * grad
            h_new = _project(h_new, N / 2.0)
        
        h_prev = h
        h = h_new
        t = t_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _nesterov_epigraph(h0, beta, n_iter=500, lr=0.01):
    """
    Nesterov accelerated gradient descent using epigraph formulation with
    auxiliary variable t. This can help escape local minima by providing
    an additional degree of freedom.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    # Initialize t at the current max overlap
    t_val = _score(h)
    
    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_mom_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_mom_new
        
        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)
        
        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)
        
        step_lr = lr
        h_new = None
        t_new_val = None
        for bt in range(20):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)
            
            # Score is the max of C_k, not t + smooth penalty
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                t_new_val = t_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            h_new = y_h - lr * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr * grad_t)
        
        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _spectral_to_h(coeffs, n_modes):
    """Convert spectral coefficients to height sequence.
    coeffs[0] = DC offset, coeffs[1:] = amplitudes for modes 1..n_modes.
    Returns h of length N clipped to [0,1] with sum adjusted to N/2.
    """
    n = N
    h = np.zeros(n)
    h[:] = coeffs[0]  # DC component
    for m in range(1, n_modes + 1):
        h += coeffs[m] * np.cos(2 * np.pi * m * np.arange(n) / n)
    return h


def _spectral_objective(coeffs, n_modes, beta):
    """Objective in spectral space: smooth max of C_k."""
    h = _spectral_to_h(coeffs, n_modes)
    cs = _linear_overlap_values(h)
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    return val


def _spectral_optimize(h_init, n_modes, beta, max_time=10.0):
    """Optimize in spectral space using scipy minimize (L-BFGS-B).
    Start from DFT of h_init projected onto first n_modes.
    """
    t0 = time.time()
    n = N
    # Initialize spectral coefficients from DFT of h_init
    H = np.fft.rfft(h_init)
    coeffs0 = np.zeros(n_modes + 1)
    coeffs0[0] = h_init.mean()  # DC
    for m in range(1, n_modes + 1):
        coeffs0[m] = 2.0 * np.real(H[m]) / n  # amplitude
    
    # Bounds: DC in [0,1], amplitudes in [-0.5, 0.5]
    bounds = [(0.0, 1.0)] + [(-0.5, 0.5)] * n_modes
    
    def obj(c):
        if time.time() - t0 > max_time:
            return 1e6
        return _spectral_objective(c, n_modes, beta)
    
    res = minimize(obj, coeffs0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                            'maxfun': 5000})
    
    h_opt = _spectral_to_h(res.x, n_modes)
    h_opt = _project(h_opt, n / 2.0)
    s_opt = _score(h_opt)
    return h_opt, s_opt


def _coordinate_descent_polish(h0, max_outer=50, max_inner=N):
    """
    Coordinate descent polish: adjust one element at a time with
    local search around current value. Uses adaptive step sizes and
    a finer grid near the current value.
    """
    h = np.array(h0, dtype=np.float64).copy()
    best_score = _score(h)
    best_h = h.copy()
    
    # Track best step size per coordinate for adaptive refinement
    best_step = np.full(N, 0.01)
    
    for outer in range(max_outer):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        for i in range(max_inner):
            if time.time() > _DEADLINE - 2:
                break
            old_val = h[i]
            step = best_step[i]
            
            # Adaptive grid: start with larger steps, refine if improvement
            deltas = [step, step/2, step/4, step/8, step/16, step/32]
            candidates = set()
            for d in deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            # Also try very fine steps
            fine_deltas = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            for d in fine_deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            candidates.add(0.0)
            candidates.add(1.0)
            
            best_candidate_val = None
            best_candidate_score = best_score - 1e-12
            
            for new_val in sorted(candidates):
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                h_test = h.copy()
                h_test[i] = new_val
                h_test = _project(h_test, N / 2.0)
                s_test = _score(h_test)
                if s_test < best_candidate_score:
                    best_candidate_score = s_test
                    best_candidate_val = new_val
                    # Update step size based on how far we moved
                    move = abs(new_val - old_val)
                    if move > 0:
                        best_step[i] = max(0.00005, move * 0.5)
            
            if best_candidate_val is not None:
                h[i] = best_candidate_val
                h = _project(h, N / 2.0)
                best_score = best_candidate_score
                best_h = h.copy()
                improved = True
    
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Initializations: diverse starting points
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N),
        0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.random.randn(N),
        0.5 + 0.2 * np.sin(6 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.cos(3 * np.pi * np.arange(N) / N),
    ]
    
    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Add random inits
    for _ in range(6):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
    
    # Phase 0: Spectral optimization (reduced dimensionality)
    # Try different numbers of modes and betas
    for n_modes in [3, 5, 8, 12]:
        if time.time() > _DEADLINE - 60:
            break
        for beta in [5.0, 10.0, 20.0]:
            if time.time() > _DEADLINE - 45:
                break
            for h_init in inits[:4]:
                if time.time() > _DEADLINE - 40:
                    break
                h_opt, s_opt = _spectral_optimize(h_init, n_modes, beta, max_time=8.0)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
    
    # Phase 1: Nesterov with adaptive beta (standard smooth max)
    beta = 10.0
    n_restarts = 4
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 40:
            break
            
        for h_init in inits:
            if time.time() > _DEADLINE - 25:
                break
                
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=350, lr=0.05)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        beta *= 1.4
        
        for h_init in inits[:3]:
            if time.time() > _DEADLINE - 25:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=250, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    # Phase 2: Epigraph Nesterov with the best solution so far
    if best_h is not None and time.time() < _DEADLINE - 20:
        beta_ep = 15.0
        for r in range(3):
            if time.time() > _DEADLINE - 15:
                break
            start_points = [best_h]
            if len(inits) > 0:
                start_points.append(inits[0])
            if len(inits) > 1:
                start_points.append(inits[1])
                
            for sp in start_points:
                if time.time() > _DEADLINE - 12:
                    break
                h_opt, s_opt = _nesterov_epigraph(sp, beta_ep, n_iter=250, lr=0.03)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            beta_ep *= 1.3
            
    # Phase 3: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_pol, s_pol = _coordinate_descent_polish(best_h, max_outer=40, max_inner=N)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()
            
    # Phase 4: Final fine-grained coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_pol2, s_pol2 = _coordinate_descent_polish(best_h, max_outer=25, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
