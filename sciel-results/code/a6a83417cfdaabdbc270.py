"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
FFT-accelerated overlap, exact Euclidean projection onto box+sum polytope,
Nesterov accelerated gradient on log-sum-exp smooth-max, and coordinate polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_DEADLINE = time.time() + 142


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


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    return val, grad


def _project(h):
    """Exact Euclidean projection onto {x : 0 <= x_i <= 1, sum(x) = N/2}."""
    h = np.asarray(h, dtype=np.float64)
    target = _TARGET
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h - mid, 0.0, 1.0))
        if s > target:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2.0
    return np.clip(h - theta, 0.0, 1.0)


def _accelerated_smooth_max(h0, beta, max_iter=500, tol=1e-10):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project(h)
    
    # Nesterov momentum
    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)
        
        # Adaptive learning rate with backtracking
        for _bt in range(10):
            h_trial = _project(h - lr * grad)
            val_trial, _ = _smooth_max_val_grad(h_trial, beta)
            if val_trial <= val + 1e-12:
                break
            lr *= 0.5
        else:
            lr = max(lr, 1e-6)
            h_trial = _project(h - lr * grad)
        
        h_new = h_trial
        lr = min(lr * 1.5, 0.5)  # grow lr if backtracking succeeded quickly
        
        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Convergence check
        if it > 10 and abs(val - val_trial) < tol * max(1.0, abs(val)):
            break
    
    return h, _score(h)


def _coordinate_polish(h, beta, max_rounds=50):
    """Coordinate descent polish: optimize each coordinate while holding others fixed."""
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    best_val = _score(h)
    
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            # Try small perturbations at coordinate i
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project(h_trial)
                s = _score(h_trial)
                if s < best_val - 1e-12:
                    h = h_trial
                    best_val = s
                    improved = True
                    break
        if not improved:
            break
    
    return h, best_val


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(12345)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    i = np.arange(N)
    
    # Diverse initial guesses
    starts = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_project(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_project(h_sin))
    
    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_project(h_2f))
    
    # Sawtooth-like
    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_project(h_saw))
    
    # Random
    for _ in range(5):
        h_r = np.random.uniform(0, 1, N)
        starts.append(_project(h_r))
    
    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)
    
    # Bisection on t with accelerated gradient feasibility
    # Lower bound: 0, Upper bound: current best
    t_low = 0.0
    t_high = best_score
    
    # For each t in bisection, run accelerated gradient to try to find feasible h
    # where smooth_max(h, beta) <= t + small_margin
    for bisect_iter in range(25):
        if time.time() > _DEADLINE - 10:
            break
        
        t_mid = (t_low + t_high) / 2.0
        beta = 50.0 + bisect_iter * 10  # Increasing beta for sharper max approximation
        
        # Try to find h with smooth_max(h, beta) <= t_mid
        # Use accelerated gradient from best_h
        if best_h is not None:
            h_candidate, s_candidate = _accelerated_smooth_max(
                best_h, beta, max_iter=300, tol=1e-12
            )
            if s_candidate < best_score:
                _update_best(h_candidate)
            
            # Also try from a perturbed start
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = _project(h_pert)
            h_candidate2, s_candidate2 = _accelerated_smooth_max(
                h_pert, beta, max_iter=200, tol=1e-12
            )
            if s_candidate2 < best_score:
                _update_best(h_candidate2)
        
        # Check if feasible: if best_score <= t_mid, go lower
        if best_score <= t_mid + 1e-6:
            t_high = t_mid
        else:
            t_low = t_mid
        
        # If gap is small, break
        if t_high - t_low < 1e-8:
            break
    
    # Final polish: high-beta accelerated gradient
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_final, s_final = _accelerated_smooth_max(best_h, beta=300.0, max_iter=400, tol=1e-14)
        if s_final < best_score:
            _update_best(h_final)
    
    # Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_coord, s_coord = _coordinate_polish(best_h, beta=300.0, max_rounds=30)
        if s_coord < best_score:
            _update_best(h_coord)
    
    # SLSQP polish on the smooth-max objective
    if best_h is not None and time.time() < _DEADLINE - 3:
        try:
            def _obj(x):
                x = _project(x)
                cs = _linear_overlap_values(x)
                m = np.max(cs)
                return m + np.log(np.sum(np.exp(50.0 * (cs - m)))) / 50.0
            
            def _obj_grad(x):
                x = _project(x)
                _, g = _smooth_max_val_grad(x, 50.0)
                return g
            
            res = minimize(
                _obj, best_h, jac=_obj_grad, method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success and res.fun < best_score:
                h_slqp = _project(res.x)
                _update_best(h_slqp)
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
