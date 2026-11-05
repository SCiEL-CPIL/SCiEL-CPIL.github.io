"""Combined program for the third autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value_from_h(h):
    """Compute the task value."""
    s = float(np.sum(h))
    if abs(s) < 1e-15:
        return 1e18
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective_and_grad(h, beta=50.0):
    """
    Smooth surrogate with analytic gradient.
    Objective: |2N * LSE(conv(h,h)) / sum(h)^2|
    where LSE is log-sum-exp approximation to max.
    """
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e18, np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = conv - m
    
    # Numerical stability
    max_shifted = np.max(shifted)
    if max_shifted > 0 and beta * max_shifted > 700:
        beta_eff = 700.0 / max_shifted
    else:
        beta_eff = beta
    
    exp_shifted = np.exp(beta_eff * shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    LSE = m + np.log(sum_exp) / beta_eff
    
    val = 2.0 * N * LSE / (s * s)
    obj = abs(val)
    
    # dLSE_dh[k] = 2 * sum_j weights[j] * h[j-k] = 2 * np.correlate(weights, h, 'valid')[k]
    dLSE_dh = 2.0 * np.correlate(weights, h, mode='valid')
    
    # d(val)/d(h_k) = 2N * (dLSE_dh[k]/s^2 - 2*LSE/s^3)
    sign_val = 1.0 if val >= 0 else -1.0
    dval_dh = 2.0 * N * (dLSE_dh / (s * s) - 2.0 * LSE / (s ** 3))
    grad = sign_val * dval_dh
    
    return obj, grad


def _spectral_init(rng, n_modes=10):
    """
    Generate initial h with flat power spectrum and suppressed DC.
    """
    freqs = rng.integers(1, N // 2, size=n_modes)
    phases = rng.uniform(0, 2 * np.pi, size=n_modes)
    amplitudes = rng.uniform(0.5, 1.5, size=n_modes)
    
    h = np.zeros(N)
    for f, ph, amp in zip(freqs, phases, amplitudes):
        h += amp * np.cos(2 * np.pi * f * np.arange(N) / N + ph)
    
    # Subtract mean to control DC
    h -= np.mean(h)
    
    # Small perturbation for diversity
    h += rng.normal(0, 0.05, N)
    
    return h


def _diverse_init(rng, max_time=10.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    x = np.arange(N)
    candidates = []
    
    # Uniform (seed)
    candidates.append(np.ones(N))
    
    # Single cosine modes with varying amplitude
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(15):
        if time.time() - t0 > max_time * 0.4:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            if time.time() - t0 > max_time * 0.6:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency cosine
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.8:
            break
        for trial in range(5):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Spectral inits
    for i in range(6):
        if time.time() - t0 > max_time * 0.95:
            break
        h = _spectral_init(rng, n_modes=8 + i)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Evaluate all candidates
    best_h = np.ones(N)
    best_val = _value_from_h(best_h)
    
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value_from_h(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    return best_h, best_val


def _local_refine(h_init, t0, deadline, max_time=25.0):
    """Local refinement with increasing beta for progressively sharper max approximation."""
    t_start = time.time()
    best_h = h_init.copy()
    best_val = _value_from_h(best_h)
    
    for beta in [30.0, 60.0, 100.0, 150.0, 300.0, 500.0]:
        elapsed = time.time() - t0
        if elapsed > deadline - 3 or time.time() - t_start > max_time:
            break
        try:
            res = minimize(
                _smooth_objective_and_grad,
                best_h,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            hard_val = _value_from_h(res.x)
            if hard_val < best_val:
                best_h = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, t0, deadline, n_starts=6, max_time=20.0):
    """Multi-start local refinement with perturbations."""
    t_start = time.time()
    best_h = h_best.copy()
    best_val = _value_from_h(best_h)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        elapsed = time.time() - t0
        if elapsed > deadline - 5 or time.time() - t_start > max_time:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                _smooth_objective_and_grad,
                h_init,
                args=(60.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value_from_h(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse initialization
    best_h, best_val = _diverse_init(rng, max_time=10.0)
    
    # Phase 2: Local refinement on the best diverse candidate
    if time.time() - t0 < deadline - 30:
        best_h, best_val = _local_refine(best_h, t0, deadline, max_time=25.0)
    
    # Phase 3: Multi-start refinement
    if time.time() - t0 < deadline - 20:
        best_h, best_val = _multi_start(best_h, t0, deadline, n_starts=6, max_time=20.0)
    
    # Phase 4: Final high-precision polish with very high beta
    if time.time() - t0 < deadline - 5:
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() - t0 > deadline - 2:
                break
            try:
                res = minimize(
                    _smooth_objective_and_grad,
                    best_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                hard_val = _value_from_h(res.x)
                if hard_val < best_val:
                    best_h = res.x.copy()
                    best_val = hard_val
            except Exception:
                pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
