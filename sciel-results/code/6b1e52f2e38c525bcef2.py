import numpy as np
import time
import scipy.optimize

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

def _phase_objective(phases, beta, deadline):
    """
    Objective for phase optimization.
    We fix the magnitude spectrum to be flat (ones) and optimize phases.
    The signal is x = irfft(ones * exp(1j*phases)).
    We want to minimize max|c_k| / S^2.
    """
    if time.time() > deadline:
        return 1e10
    
    n = N
    size = 1
    while size < 2 * n - 1:
        size *= 2
        
    # Build spectrum
    # phases length should be size//2 + 1
    # But we only need to optimize the first N-1 or so? 
    # Actually, for linear convolution of length N, we need the full spectrum of the zero-padded signal.
    # The zero-padded signal has length at least 2N-1.
    # Its FFT has size >= 2N-1.
    
    # Let's just use the full size.
    # Constraint: x must be real.
    # irfft handles this.
    
    # DC component (index 0) and Nyquist (index size//2 if size even) must be real.
    # So their phases must be 0 or pi.
    # We can fix them to 0 for simplicity, or allow sign flip.
    # Let's fix DC phase to 0 to keep sum positive (or near it).
    # Actually, sum x = S. If S is close to 0, metric is bad.
    # We should probably optimize DC amplitude too? 
    # The prompt says "fixing the input sequence's power spectrum to a constant magnitude".
    # This implies we fix |X[k]| = 1 for all k.
    # Then x = irfft(exp(1j*phases)).
    # Sum x = X[0] = exp(1j*0) = 1 (if phase 0 is 0).
    # So S=1.
    
    spec = np.exp(1j * phases)
    spec[0] = 1.0  # Fix DC
    if size % 2 == 0:
        spec[-1] = 1.0  # Fix Nyquist
        
    x_full = np.fft.irfft(spec, size)
    x = x_full[:n]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
        
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _optimize_phases(phases_init, deadline, beta_start=5.0, beta_end=50.0, n_steps=3):
    """
    Optimize phases using L-BFGS-B on the phase vector.
    """
    phases = phases_init.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
            
        t_anneal = step / max(1, n_steps - 1)
        beta = beta_start + (beta_end - beta_start) * t_anneal
        
        def obj(p):
            return _phase_objective(p, beta, deadline)
        
        try:
            result = scipy.optimize.minimize(
                obj,
                phases,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8},
                tol=1e-10
            )
            phases = result.x
        except Exception:
            pass
            
    return phases

def _phases_to_signal(phases, n):
    size = 1
    while size < 2 * n - 1:
        size *= 2
    spec = np.exp(1j * phases)
    spec[0] = 1.0
    if size % 2 == 0:
        spec[-1] = 1.0
    x_full = np.fft.irfft(spec, size)
    return x_full[:n]

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
        
        # Efficient computation using FFT? 
        # Actually, for N=400, a loop over active lags (approx 40) and vector ops is fast enough.
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
    
    # 1. Random phase flat spectrum signals (good spectral spread)
    for seed in [1, 2, 3, 4, 5]:
        x = _generate_random_phase_signal(N, seed)
        # Add small DC to avoid zero sum if needed, though random phase usually has non-zero sum
        candidates.append(x)
        
    # 2. Sum of sines
    for seed in [10, 11, 12]:
        x = _generate_sine_sum(N, K=5, seed=seed)
        candidates.append(x)
        
    # 3. Random noise
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # 4. Binary +/-1 (Barker-like start)
    candidates.append(rng.choice([-1, 1], N))
    
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
            
    # Phase 3: Phase-only optimization
    # This is a different parameterization.
    # We fix |X[k]| = 1 and optimize phases.
    # This ensures spectral flatness.
    
    size = 1
    while size < 2 * N - 1:
        size *= 2
    n_phase_params = size // 2 + 1
    
    for seed in [100, 101, 102, 103, 104]:
        if time.time() > deadline - 10.0:
            break
            
        rng = np.random.default_rng(seed)
        phases_init = rng.uniform(-np.pi, np.pi, n_phase_params)
        
        # Optimize phases
        phases_opt = _optimize_phases(phases_init, deadline)
        
        # Convert to time domain
        x_phase = _phases_to_signal(phases_opt, N)
        
        # Evaluate
        sc_phase = _metric_offdiag(x_phase)
        update_best(x_phase, sc_phase)
        
        # Polish the result in time domain
        if time.time() < deadline - 5.0:
            x_pol, sc_pol = _polish_subgradient(x_phase, deadline)
            update_best(x_pol, sc_pol)
            
    # If we have time, do a final polish on the best one
    if best_x is not None and time.time() < deadline - 2.0:
        x_final, sc_final = _polish_subgradient(best_x, deadline, max_iters=20)
        update_best(x_final, sc_final)
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
