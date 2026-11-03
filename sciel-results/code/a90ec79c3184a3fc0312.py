"""Erdos minimum overlap: smooth-surrogate optimization with bisection on t.
Combines the epigraph formulation with a bisection-based feasibility check
using projected gradient descent (with Nesterov acceleration) on the smooth
log-sum-exp surrogate. This avoids SLSQP's tendency to get stuck in local
minima by systematically narrowing the feasible t range. Multiple diverse
starts and annealed beta are used.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    
    # If sum is already correct and bounds are satisfied, return
    if abs(np.sum(h) - target) < 1e-12 and np.min(h) >= -1e-12 and np.max(h) <= 1 + 1e-12:
        return np.clip(h, 0.0, 1.0)
        
    # We need to find tau such that sum(max(0, min(1, h_i - tau))) = target
    # Let f(tau) = sum(max(0, min(1, h_i - tau)))
    # f is continuous, non-increasing.
    # If f(0) < target, we need to increase some h_i. This implies tau < 0.
    # If f(-1) > target, we need to decrease some h_i. This implies tau > 0.
    
    # Binary search for tau
    # Range for tau:
    # If we set tau very negative, all h_i - tau > 1, so min(1, ...) = 1, sum = N.
    # If we set tau very positive, all h_i - tau < 0, so max(0, ...) = 0, sum = 0.
    # So tau is roughly in [min(h) - 1, max(h) + 1] or wider.
    
    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0
    
    # Check endpoints
    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))
    
    if calc_sum(lo) < target:
        # Need even lower tau? No, f is decreasing. If f(lo) < target, we need smaller tau?
        # Wait. f(tau) decreases as tau increases.
        # If f(lo) < target, we need f(tau) to be larger, so we need smaller tau.
        # But lo is min(h)-2. If min(h) is 0, lo=-2. h_i - (-2) = h_i+2. Clip to 1. Sum is N.
        # So f(lo) should be N.
        # Let's just widen bounds if needed.
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-12:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)
    
    # Final exact correction to handle floating point drift
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        # Distribute to elements not at bounds
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            # Fallback: adjust one element
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s  # gradient w.r.t. x
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh). Vectorized."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _projected_gradient_step(h, grad, lr):
    """One projected gradient step: h_new = proj(h - lr * grad)."""
    h_new = h - lr * grad
    return _project(h_new)


def _nesterov_accelerated_gradient(h0, beta, n_steps=2000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate with backtracking.
    Returns best h found and its score."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]
    
    if lr is None:
        # Initial guess for step size
        lr = 1.0 / (beta * 10.0) 
    
    momentum = 0.9
    t_k = 1.0
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        # Nesterov lookahead
        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        
        # Backtracking line search for step size
        # We want f(h_new) <= f(y) + <grad_y, h_new - y> + (1/(2*lr)) ||h_new - y||^2
        # But with projection, this is tricky. Simpler: just check if f decreases.
        # Or use a standard backtracking: f(y - alpha * grad) < f(y) - c * alpha * ||grad||^2
        # However, projection changes the geometry.
        # Let's try a simpler adaptive strategy:
        # Start with current lr. If f(h_new) > f(h) + 1e-10, reduce lr.
        
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]
        
        # Backtracking
        max_backtracks = 10
        backtracks = 0
        while val_new > best_val + 1e-12 and backtracks < max_backtracks:
            lr *= 0.8
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
            backtracks += 1
            
        # If we still didn't improve, accept anyway but reduce lr further to stabilize
        if val_new > best_val + 1e-12:
            lr *= 0.5
            
        # Update momentum
        # Standard Nesterov: t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        # y_{k+1} = h_{k+1} + (t_k - 1)/t_{k+1} * (h_{k+1} - h_k)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new
        
        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()
            
    return best_h, _score(best_h)


def _bisection_check(t_target, h0, beta, n_steps=1500, time_budget=8.0):
    """Check if there exists h with max(C(h)) <= t_target using smooth surrogate.
    Minimize S_beta(C(h) - t_target) and check if result is close to 0."""
    t_start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_obj = float('inf')
    
    # Use projected gradient with a modified objective
    def _obj_grad_t(h):
        cs = _overlap_values(h)
        shifted = cs - t_target
        val, g_c = _smooth_max_and_grad(shifted, beta)
        # Gradient w.r.t. h
        g_h = np.zeros(N)
        factor = -(2.0 / N)
        for k in range(N):
            gc = g_c[k]
            if gc == 0.0:
                continue
            if k == 0:
                g_h += factor * gc * (2.0 * h)
            else:
                g_h[k:] += factor * gc * h[:N - k]
                g_h[:N - k] += factor * gc * h[k:]
        return val, g_h
    
    # Nesterov accelerated projected gradient
    lr = 1.0 / (beta * 2.0)
    momentum = 0.9
    y = h.copy()
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        val_y, grad_y = _obj_grad_t(y)
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        y = h_new + (momentum / t_k) * (h_new - h)
        y = _project(y)
        h = h_new
        
        val_h, _ = _obj_grad_t(h)
        if val_h < best_obj:
            best_obj = val_h
            best_h = h.copy()
        
        if step > 10 and val_h > best_obj + 1e-10:
            lr *= 0.95
    
    # Check if we found a feasible solution
    cs = _overlap_values(best_h)
    max_c = np.max(cs)
    feasible = max_c <= t_target + 1e-8
    return feasible, best_h, max_c


def _refine_smooth(h0, beta, n_max_iter=3000, time_budget=25.0):
    """Refine h using Nesterov accelerated projected gradient on the smooth surrogate."""
    h0 = _project(h0.copy())
    h_opt, s_opt = _nesterov_accelerated_gradient(h0, beta, n_steps=n_max_iter, time_budget=time_budget)
    return h_opt, s_opt


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def _bisection_optimize(h0, beta, t_lo, t_hi, n_bisect=15, time_budget=30.0):
    """Bisection on t to find the minimum feasible max overlap.
    Returns best h and its score."""
    t_start = time.time()
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Ensure t_hi is feasible
    for _ in range(5):
        if time.time() - t_start > time_budget:
            break
        feasible, h_try, max_c = _bisection_check(t_hi, h0, beta, n_steps=800, 
                                                   time_budget=min(5.0, time_budget - (time.time() - t_start)))
        if feasible:
            break
        t_hi += 0.05
        if t_hi > 1.0:
            t_hi = 1.0
    
    # Bisection
    for bisect_iter in range(n_bisect):
        if time.time() - t_start > time_budget:
            break
        
        t_mid = (t_lo + t_hi) / 2.0
        remaining = time_budget - (time.time() - t_start)
        if remaining < 3.0:
            break
        
        feasible, h_try, max_c = _bisection_check(t_mid, h0, beta, 
                                                   n_steps=1200, time_budget=min(8.0, remaining - 1.0))
        
        if feasible:
            t_hi = t_mid
            if max_c < best_score:
                best_score = max_c
                best_h = h_try.copy()
        else:
            t_lo = t_mid
        
        # Update h0 to the best found so far for next iteration
        h0 = best_h.copy()
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 50:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with Nesterov accelerated gradient
    # Use annealed beta for sharper max approximation
    beta_list = [10.0, 50.0, 100.0, 300.0, 600.0]
    
    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break
        
        time_per_beta = min(15.0, (remaining - 10.0) / 2.0)
        if time_per_beta < 5.0:
            break
        
        h_ref, s_ref = _refine_smooth(best_h, beta=beta, n_max_iter=3000, time_budget=time_per_beta)
        _update(h_ref)
        
        # Add perturbation of best for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 30:
            h_pert = _perturb(best_h, rng, scale=0.02)
            h_ref2, s_ref2 = _refine_smooth(h_pert, beta=beta, n_max_iter=2000, time_budget=min(10.0, _DEADLINE - time.time() - 10))
            _update(h_ref2)

    # Phase 5: Final high-precision polish using SciPy L-BFGS-B
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            from scipy.optimize import minimize
            
            # Use a high beta for the polish to approximate the max closely
            beta_polish = 1000.0
            
            def _polish_obj(x):
                val, _ = _smooth_obj_and_grad(x, beta_polish)
                return val
            
            def _polish_grad(x):
                _, g = _smooth_obj_and_grad(x, beta_polish)
                return g
            
            bounds = [(0.0, 1.0) for _ in range(N)]
            # Note: L-BFGS-B does not handle equality constraints like sum(x)=N/2 directly.
            # We rely on the fact that the optimum of the smooth surrogate might naturally
            # satisfy the constraint or be close, but this is risky.
            # Better: Use the projected gradient result as a strong start and run
            # a few more Nesterov steps with very high beta.
            
            # Let's stick to Nesterov with very high beta for the final polish
            # to avoid constraint violations.
            h_polish, s_polish = _refine_smooth(best_h, beta=beta_polish, n_max_iter=2000,
                                                time_budget=min(10.0, remaining))
            _update(h_polish)
            
            # One more pass with even higher beta
            if time.time() - t0 < _DEADLINE - 5:
                remaining2 = _DEADLINE - time.time() - 2.0
                if remaining2 > 2.0:
                    h_polish2, s_polish2 = _refine_smooth(best_h, beta=2000.0, n_max_iter=1000,
                                                          time_budget=min(5.0, remaining2))
                    _update(h_polish2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
