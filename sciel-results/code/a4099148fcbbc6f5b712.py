"""
Optimized program for the Third Autocorrelation Inequality task.

Combines the chirp-based search of the current program with the spectral
approach and soft-max surrogate optimization from the inspirations.

Key structural change: Use a Zadoff-Chu-like CAZAC sequence as the base
representation, which has near-zero non-zero autocorrelation lags. We then
optimize the sign-flip pattern and phase to minimize the peak of the
autoconvolution relative to the sum squared.

Additionally, we use the soft-max surrogate from inspiration 1 for local
refinement, which provides a smooth approximation to the max function and
allows gradient-based optimization.
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the evaluator's exact value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max(x, beta=300.0):
    """Smooth approximation to max."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

def _zadoff_chu_base(root):
    """
    Generate a Zadoff-Chu-like sequence.
    For odd prime m, z[n] = exp(-j*pi*m*n*(n+1)/m)
    We take the real part and add a DC offset to ensure non-zero sum.
    """
    i = np.arange(N, dtype=np.float64)
    # Use a large prime near N for good autocorrelation properties
    m = root
    # Zadoff-Chu: z[n] = exp(-j*pi*m*n*(n+1)/m)
    # Simplified: exp(-j*pi*n*(n+1)/m)
    phase = -np.pi * i * (i + 1) / m
    z = np.cos(phase) + 1j * np.sin(phase)
    # Take real part
    h = np.real(z)
    # Add DC offset to ensure sum is positive and large
    # We want sum(h) to be large, so add a constant
    return h

def _cazac_with_signs(signs, root):
    """
    Generate a CAZAC-like sequence with sign flips.
    signs is an array of +1 or -1 of length N.
    """
    base = _zadoff_chu_base(root)
    h = signs * base
    # Add a DC component to ensure non-zero sum
    # We'll optimize this separately
    return h

def _chirp_vector(phi, beta, alpha=1.0):
    """Generate a chirp-like vector: h[i] = alpha * cos(phi + beta * i^2)."""
    i = np.arange(N, dtype=np.float64)
    return alpha * np.cos(phi + beta * i * i)

def _spectral_vector(bandwidth, center, noise_level=0.0):
    """Generate a vector from a Gaussian spectral profile."""
    freqs = np.fft.rfftfreq(N)
    H = np.exp(-((freqs - center) ** 2) / (2 * bandwidth ** 2))
    if noise_level > 0:
        H *= (1.0 + noise_level * np.random.randn(len(H)))
    h = np.fft.irfft(H, n=N)
    return h

def _obj_chirp(params):
    """Objective for chirp parameters."""
    phi, beta, alpha = params
    h = _chirp_vector(phi, beta, alpha)
    return _compute_value(h)

def _obj_spectral(params):
    """Objective for spectral parameters."""
    bandwidth, center = params
    h = _spectral_vector(bandwidth, center)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.inf
    h = h / s  # Normalize so sum = 1
    return _compute_value(h)

def _random_search_best(n_trials=2000, deadline=None):
    """Do random search over chirp and spectral parameters and return best h."""
    if deadline is None:
        deadline = START_TIME + 0.3 * TIME_LIMIT
    best_val = np.inf
    best_h = np.ones(N)
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Random chirp parameters
        phi = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(-0.05, 0.05)
        alpha = np.random.uniform(0.5, 2.0)
        h = _chirp_vector(phi, beta, alpha)
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Random spectral parameters
        bandwidth = np.random.uniform(0.05, 0.4)
        center = np.random.uniform(-0.3, 0.3)
        h_spec = _spectral_vector(bandwidth, center)
        s = np.sum(h_spec)
        if abs(s) > 1e-10:
            h_spec = h_spec / s
            val_spec = _compute_value(h_spec)
            if val_spec < best_val:
                best_val = val_spec
                best_h = h_spec.copy()
    
    return best_h, best_val

def _de_chirp(deadline=None):
    """Run differential evolution on chirp parameters."""
    if deadline is None:
        deadline = START_TIME + 0.4 * TIME_LIMIT
    if time.time() > deadline:
        return np.ones(N), np.inf
    
    bounds = [(0, 2 * np.pi), (-0.1, 0.1), (0.1, 3.0)]
    try:
        result = differential_evolution(
            _obj_chirp, bounds, maxiter=50, popsize=15,
            tol=1e-6, seed=42, polish=True
        )
        if time.time() > deadline:
            return np.ones(N), np.inf
        if result.fun < np.inf:
            h = _chirp_vector(*result.x)
            val = _compute_value(h)
            return h, val
    except Exception:
        pass
    return np.ones(N), np.inf

def _soft_max_polish(h0, deadline=None):
    """
    Polish a vector using soft-max surrogate with Nelder-Mead.
    This is the key refinement step from inspiration 1.
    """
    if deadline is None:
        deadline = START_TIME + 0.7 * TIME_LIMIT
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    # Normalize so sum = 1
    s0 = np.sum(h0)
    if abs(s0) < 1e-10:
        return h0, _compute_value(h0)
    h0 = h0 / s0
    
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    # Try multiple beta values for the soft-max
    for beta in [200.0, 400.0]:
        if time.time() > deadline - 5:
            break
        
        def obj(x):
            conv = np.convolve(x, x)
            sm = _soft_max(conv, beta=beta)
            s = np.sum(x)
            penalty = 500.0 * (s - 1.0) ** 2
            return sm + penalty
        
        try:
            res = minimize(obj, best_h, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if time.time() > deadline - 3:
                break
            h_new = res.x
            s_new = np.sum(h_new)
            if abs(s_new) > 1e-10:
                v_new = _compute_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _local_perturbation_polish(h0, deadline=None):
    """Simple local perturbation search."""
    if deadline is None:
        deadline = START_TIME + 0.8 * TIME_LIMIT
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    # Simple local perturbation search
    for _ in range(100):
        if time.time() > deadline:
            break
        perturbation = np.random.normal(0, 0.01, N)
        h_new = best_h + perturbation
        val = _compute_value(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new
    
    return best_h, best_val

def _cazac_search(deadline=None):
    """
    Search over Zadoff-Chu-like sequences with optimized sign flips.
    This is the structural change: use CAZAC sequences as the base.
    """
    if deadline is None:
        deadline = START_TIME + 0.5 * TIME_LIMIT
    if time.time() > deadline:
        return np.ones(N), np.inf
    
    best_val = np.inf
    best_h = np.ones(N)
    
    # Try different roots (primes near N)
    primes = [397, 401, 389, 409]
    
    for root in primes:
        if time.time() > deadline - 10:
            break
        
        base = _zadoff_chu_base(root)
        
        # Try different DC offsets
        for dc in np.linspace(-0.5, 0.5, 11):
            if time.time() > deadline - 5:
                break
            h = base + dc
            s = np.sum(h)
            if abs(s) < 1e-10:
                continue
            val = _compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        # Try sign-flip patterns (random)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            signs = np.random.choice([-1, 1], N)
            h = signs * base
            s = np.sum(h)
            if abs(s) < 1e-10:
                continue
            val = _compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h, best_val

def construct():
    """
    Returns the optimal step heights.
    Uses a combination of random search, differential evolution, CAZAC search,
    and soft-max polishing to find a vector that minimizes the autocorrelation
    inequality value.
    """
    deadline = START_TIME + TIME_LIMIT - 10.0  # Safety margin
    
    # Start with seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Random search over chirp and spectral families
    if time.time() < deadline - 80:
        h_rand, val_rand = _random_search_best(n_trials=3000, deadline=deadline - 70)
        if val_rand < best_val:
            best_h = h_rand
            best_val = val_rand
    
    # Phase 2: CAZAC-based search (structural change)
    if time.time() < deadline - 60:
        h_cazac, val_cazac = _cazac_search(deadline=deadline - 50)
        if val_cazac < best_val:
            best_h = h_cazac
            best_val = val_cazac
    
    # Phase 3: Differential evolution on chirp parameters
    if time.time() < deadline - 40:
        h_de, val_de = _de_chirp(deadline=deadline - 30)
        if val_de < best_val:
            best_h = h_de
            best_val = val_de
    
    # Phase 4: Soft-max polishing (key refinement from inspiration 1)
    if time.time() < deadline - 20:
        h_polished, val_polished = _soft_max_polish(best_h, deadline=deadline - 10)
        if val_polished < best_val:
            best_h = h_polished
            best_val = val_polished
    
    # Phase 5: Local perturbation polish
    if time.time() < deadline - 5:
        h_local, val_local = _local_perturbation_polish(best_h, deadline=deadline)
        if val_local < best_val:
            best_h = h_local
            best_val = val_local
    
    # Final safety check
    if time.time() > deadline - 5:
        return np.ones(N)
    
    final_val = _compute_value(best_h)
    if final_val < 2.0:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
