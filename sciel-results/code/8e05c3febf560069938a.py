import numpy as np
import time
import scipy.optimize
from scipy.optimize import minimize, minimize_scalar

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    if len(abs_c) == 0:
        return 0.0
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    # Numerically stable log-sum-exp
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _objective_soft(x, beta, start_time, deadline):
    """Objective function for L-BFGS-B with time guard."""
    if time.time() > deadline:
        return 1e10
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _generate_random_phase_signal(n, seed):
    """Generate a signal with flat spectrum and random phases."""
    rng = np.random.default_rng(seed)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
    mag = np.ones(size // 2 + 1)
    spectrum = mag * np.exp(1j * phases)
    spectrum[0] = mag[0]  # DC real
    if size % 2 == 0:
        spectrum[-1] = mag[-1]  # Nyquist real
    x_full = np.fft.irfft(spectrum, size)
    x = x_full[:n]
    return x

def _generate_sine_sum(n, K, seed):
    """Generate a signal as sum of K random sines."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = np.zeros(n)
    for _ in range(K):
        f = rng.uniform(1, n/2)
        p = rng.uniform(0, 2*np.pi)
        a = rng.uniform(0.5, 1.5)
        x += a * np.sin(2 * np.pi * f * t / n + p)
    return x

def _zadoff_chu_real(n, u, N_local):
    """Real part of Zadoff-Chu-like sequence: cos(pi * u * n * (n+1) / N)."""
    return np.cos(np.pi * u * n * (n + 1) / N_local)

def _optimize_chirp_param(n, N_local, u_range=(-5.0, 5.0)):
    """Find best chirp parameter u via grid + local refine."""
    u_grid = np.linspace(u_range[0], u_range[1], 50)
    best_u = u_grid[0]
    best_score = float('inf')
    for u in u_grid:
        x = _zadoff_chu_real(n, u, N_local)
        s = _metric_offdiag(x)
        if s < best_score:
            best_score = s
            best_u = u
    
    delta = (u_range[1] - u_range[0]) / 50
    try:
        res = minimize_scalar(
            lambda u: _metric_offdiag(_zadoff_chu_real(n, u, N_local)),
            bounds=(best_u - delta, best_u + delta),
            method='bounded',
            options={'xatol': 1e-6}
        )
        if res.fun < best_score:
            best_u = res.x
            best_score = res.fun
    except Exception:
        pass
    
    return best_u, best_score

def _refine_lbfgs(x_init, deadline, beta_start=10.0, beta_end=100.0, n_anneal_steps=5):
    """
    Refine x_init using L-BFGS-B with annealed soft-max beta.
    """
    x = x_init.copy()
    current_beta = beta_start
    
    # Anneal beta in a few stages
    for step in range(n_anneal_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Interpolate beta
        t_anneal = step / max(1, n_anneal_steps - 1)
        beta = beta_start + (beta_end - beta_start) * t_anneal
        
        def obj(x):
            return _objective_soft(x, beta, time.time(), deadline)
        
        try:
            result = scipy.optimize.minimize(
                obj,
                x,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8},
                tol=1e-10
            )
            x = result.x
        except Exception:
            pass

    return x

def _polish_subgradient(x, deadline, max_iters=50):
    """
    Polish using subgradient descent on the exact max |c_k|.
    """
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    for i in range(max_iters):
        if time.time() > deadline - 1.0:
            break
            
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
            
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        
        # Find active lags (top 10%)
        threshold = np.percentile(abs_c, 90)
        active_mask = abs_c > threshold
        if not np.any(active_mask):
            break
            
        # Subgradient: sum of gradients of active terms
        # d(c_k)/dx_j = x_{j+k} + x_{j-k}
        # grad = sum_{k in active} sign(c_k) * (x shifted by k + x shifted by -k)
        
        grad = np.zeros_like(x)
        active_indices = np.where(active_mask)[0]
        
        for k_idx in active_indices:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            # x[j+k]
            if k < N:
                grad[:N-k] += sign_c * x[k:]
            # x[j-k]
            if k < N:
                grad[k:] += sign_c * x[:N-k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
            
        # Step size
        # Use a simple adaptive step
        step_size = 0.01 * (1.0 - i/max_iters) + 0.001
        x_new = x - step_size * grad / gnorm
        
        # Renormalize to keep sum stable? 
        # The metric is scale invariant, but L-BFGS might drift. 
        # Let's just evaluate.
        sc_new = _metric_offdiag(x_new)
        
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            # Reduce step
            step_size *= 0.5
            x_new = x - step_size * grad / gnorm
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new

    return best_x, best_score

def _local_bit_flip_fast(x0, max_passes=20, deadline=None):
    """Local optimization by flipping signs using incremental autoconvolution updates."""
    x = x0.copy().astype(float)
    n = len(x)
    c = _autoconv_lin(x)
    S = np.sum(x)
    if abs(S) < 1e-12:
        return x, float('inf')
    
    current_max_c = np.max(np.abs(c[1:]))
    current_metric = current_max_c / (S * S)
    if not np.isfinite(current_metric):
        return x, current_metric

    rng = np.random.default_rng(12345)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx = rng.permutation(n)
        for j in idx:
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = x[j]
            new_val = -old_val
            dx = new_val - old_val
            
            S_new = S + dx
            if abs(S_new) < 1e-12:
                continue
                
            affected_k = range(max(0, j), min(2*n-1, j+n))
            old_c_vals = c[affected_k].copy()
            
            for k in affected_k:
                if k == 2*j:
                    delta = dx * x[j]
                else:
                    idx_x = k - j
                    if 0 <= idx_x < n:
                        delta = 2 * dx * x[idx_x]
                    else:
                        delta = 0
                c[k] += delta
            
            new_max_c = np.max(np.abs(c[1:]))
            new_metric = new_max_c / (S_new * S_new)
            
            if new_metric < current_metric - 1e-12:
                x[j] = new_val
                S = S_new
                current_max_c = new_max_c
                current_metric = new_metric
                improved = True
            else:
                c[affected_k] = old_c_vals

        if not improved:
            break
            
    return x, current_metric

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
            
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Zadoff-Chu chirp family (from inspiration)
    n_arr = np.arange(N)
    u_best, s_chirp = _optimize_chirp_param(n_arr, N, u_range=(-5.0, 5.0))
    x_chirp = _zadoff_chu_real(n_arr, u_best, N)
    candidates.append(x_chirp)
    
    # Try a few more chirp parameters with different u ranges
    for u_range in [(-1.0, 1.0), (-2.0, 2.0), (0.5, 3.0)]:
        if time.time() > deadline - 40:
            break
        u_b, _ = _optimize_chirp_param(n_arr, N, u_range=u_range)
        x_c = _zadoff_chu_real(n_arr, u_b, N)
        candidates.append(x_c)
    
    # 2. Random phase flat spectrum signals (good spectral spread)
    for seed in [1, 2, 3, 4, 5]:
        x = _generate_random_phase_signal(N, seed)
        candidates.append(x)
        
    # 3. Sum of sines
    for seed in [10, 11, 12]:
        x = _generate_sine_sum(N, K=5, seed=seed)
        candidates.append(x)
        
    # 4. Random noise
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # 5. Binary +/-1 (Barker-like start)
    candidates.append(rng.choice([-1, 1], N))
    
    # 6. Block patterns
    for block_size in [10, 20, 50]:
        if time.time() > deadline - 30:
            break
        x = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                x[i] = 1.0
            else:
                x[i] = -1.0
        x[0] += 0.1  # ensure non-zero sum
        candidates.append(x)
    
    # Phase 2: Refine each candidate
    for i, x_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        # Quick score check
        sc_init = _metric_offdiag(x_init)
        update_best(x_init, sc_init)
        
        # Refine with L-BFGS
        x_ref = _refine_lbfgs(x_init, deadline)
        sc_ref = _metric_offdiag(x_ref)
        update_best(x_ref, sc_ref)
        
        # Polish with subgradient
        if time.time() < deadline - 5.0:
            x_pol, sc_pol = _polish_subgradient(x_ref, deadline)
            update_best(x_pol, sc_pol)
            
        # Bit-flip polish (from inspiration)
        if time.time() < deadline - 3.0:
            x_flip, sc_flip = _local_bit_flip_fast(x_pol, max_passes=10, deadline=deadline)
            update_best(x_flip, sc_flip)
            
    # If we have time, do a final polish on the best one
    if best_x is not None and time.time() < deadline - 2.0:
        x_final, sc_final = _polish_subgradient(best_x, deadline, max_iters=20)
        update_best(x_final, sc_final)
        
        # Final bit-flip
        if time.time() < deadline - 1.0:
            x_final2, sc_final2 = _local_bit_flip_fast(x_final, max_passes=5, deadline=deadline)
            update_best(x_final2, sc_final2)
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
