"""Combined program for third autocorrelation inequality optimization.

Uses unitary-transform initialization (flat power spectrum via Parseval)
combined with soft-max surrogate optimization and spectral-aware refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:2 * n - 1]
    return conv

def _evaluate(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = m + np.log(sum_exp) / beta
    return soft_max, weights

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    Uses FFT cross-correlation for O(N log N) computation.
    """
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    return 2.0 * N * (grad_sm / (s * s) - 2.0 * soft_max * np.ones_like(h) / (s * s * s))

def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of FFT coefficients)."""
    H = np.fft.rfft(h)
    power = np.abs(H) ** 2
    total = np.sum(power)
    if total < 1e-10:
        return 0.0
    max_p = np.max(power)
    avg_p = total / len(power)
    return lam * (max_p / (avg_p + 1e-10) - 1.0)

def _unitary_initializations(rng, count=20):
    """
    Generate initializations using random unitary transformations of sparse impulses.
    This enforces a flat power spectrum (via Parseval) which helps minimize
    the autoconvolution peak-to-average ratio.
    """
    inits = []
    for _ in range(count):
        h_sparse = np.zeros(N)
        num_impulses = rng.integers(5, 25)
        idxs = rng.choice(N, size=num_impulses, replace=False)
        h_sparse[idxs] = rng.choice([-1.0, 1.0], size=num_impulses)
        
        H = np.fft.rfft(h_sparse)
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(H)))
        H_rot = H * phases
        h_rot = np.fft.irfft(H_rot, n=N)
        
        mean = np.mean(h_rot)
        h_rot = h_rot - mean + 1.0
        
        inits.append(h_rot)
    return inits

def _diverse_initializations(rng):
    """Generate a diverse set of initializations combining unitary and structured patterns."""
    inits = []
    i = np.arange(N)
    
    # 1. Seed
    inits.append(np.ones(N).copy())
    
    # 2. Unitary-transformed sparse impulses (key structural idea)
    inits.extend(_unitary_initializations(rng, count=15))
    
    # 3. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2 * np.pi * k * i / N)
            inits.append(h0.copy())
    
    # 4. Comb patterns with small noise
    for period in [2, 3, 4, 5, 6, 8, 10, 12, 16]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 = h0 + 0.1 * rng.standard_normal(N)
        inits.append(h0.copy())
    
    # 5. Higher frequency waves
    for freq in [3, 7, 11, 15, 20, 30]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h0 = 1.0 + np.sin(t) + 0.5 * np.cos(2 * t)
        h0 += 0.1 * rng.standard_normal(N)
        inits.append(h0.copy())
    
    # 6. Block patterns
    for block_size in [8, 16, 25, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b * block_size:(b + 1) * block_size] = rng.standard_normal()
        h0 += 1.0
        inits.append(h0.copy())
    
    # 7. Random normal
    for _ in range(5):
        h0 = rng.standard_normal(N) + 1.0
        inits.append(h0)
    
    return inits

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    remaining = deadline - time.time()
    
    if remaining < 3.0:
        return h_curr
    
    # Phase 1: Moderate beta with spectral penalty
    if remaining > 15.0:
        try:
            def obj1(x):
                return _objective_soft(x, beta=30.0) + _spectral_penalty(x, lam=0.1)
            
            def grad1(x):
                g = _objective_soft_grad(x, beta=30.0)
                # Approximate spectral penalty gradient via finite difference or skip
                return g
            
            res = minimize(obj1, h_curr, method='L-BFGS-B',
                           jac=grad1,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            if res.fun < _evaluate(h_curr):
                h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty
    if deadline - time.time() > 10.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(60.0,), jac=_objective_soft_grad,
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9})
            if _evaluate(res.x) < _evaluate(h_curr):
                h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Very high beta - peak polishing
    if deadline - time.time() > 5.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(150.0,), jac=_objective_soft_grad,
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if _evaluate(res.x) < _evaluate(h_curr):
                h_curr = res.x
        except Exception:
            pass
    
    # Phase 4: Extremely high beta for final polish
    if deadline - time.time() > 2.5:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(300.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11})
            if _evaluate(res.x) < _evaluate(h_curr):
                h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s before 150s kill)
    
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate diverse initializations
    initializations = _diverse_initializations(rng)
    
    # Optimize each initialization with time-aware refinement
    for init_idx, h0 in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Estimate time per refinement
        n_remaining = len(initializations) - init_idx
        time_per_refine = max(2.0, (remaining - 5.0) / max(n_remaining, 1))
        
        if time_per_refine < 3.0:
            # Not enough time for full refinement, just evaluate
            val = _evaluate(h0)
            if val < best_val:
                best_val = val
                best_h = h0.copy()
            continue
        
        # Refine this initialization
        h_refined = _refine(h0, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 8.0:
        for beta in [500.0, 1000.0]:
            if time.time() > deadline - 4.0:
                break
            try:
                res = minimize(_objective_soft, best_h, method='L-BFGS-B',
                               args=(beta,), jac=_objective_soft_grad,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                val = _evaluate(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
