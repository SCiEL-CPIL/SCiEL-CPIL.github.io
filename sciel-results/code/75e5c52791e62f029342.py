"""Signed autocorrelation max-abs minimization via multi-sine spectral parameterization + DE + local search."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 150.0
_MARGIN = 15.0  # safety margin


def _autoconv(a):
    """Full linear autoconvolution of a (length N)."""
    F = np.fft.rfft(a, n=2 * N - 1)
    return np.fft.irfft(F * F, n=2 * N - 1)


def _score(a):
    """Upper bound on C3': max|autoconv| / (sum a)^2."""
    ac = _autoconv(a)
    s = a.sum()
    if abs(s) < 1e-12:
        return np.max(np.abs(ac))
    return np.max(np.abs(ac)) / (s * s)


def _sign_vector_from_sines(freqs, amps, phases):
    """Construct a sign vector from a multi-sine model.
    
    x[i] = sign( sum_k amps[k] * sin(2*pi*freqs[k]*i/N + phases[k]) )
    """
    t = np.arange(N)
    signal = np.zeros(N)
    for k in range(len(freqs)):
        signal += amps[k] * np.sin(2 * np.pi * freqs[k] * t / N + phases[k])
    s = np.sign(signal)
    s[s == 0] = 1.0
    return s


def _evaluate_params(params):
    """Evaluate a parameter vector. Returns (score, sign_vector)."""
    n_sines = int(params[0])
    n_sines = max(1, min(n_sines, 12))
    
    freqs = params[1:1 + n_sines]
    amps = params[1 + n_sines:1 + 2 * n_sines]
    phases = params[1 + 2 * n_sines:1 + 3 * n_sines]
    
    s = _sign_vector_from_sines(freqs, amps, phases)
    v = _score(s)
    return v, s


def _local_search(s, max_rounds=8):
    """Flip-based local search to reduce peak autoconvolution."""
    best_s = s.copy()
    best_v = _score(best_s)
    
    for _round in range(max_rounds):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        ac = _autoconv(best_s)
        peak = np.max(np.abs(ac))
        s_sum = best_s.sum()
        denom = s_sum * s_sum
        
        for i in range(N):
            if time.time() > _DEADLINE - _MARGIN:
                break
            best_s[i] = -best_s[i]
            ac2 = _autoconv(best_s)
            peak2 = np.max(np.abs(ac2))
            s_sum2 = best_s.sum()
            denom2 = s_sum2 * s_sum2
            v2 = peak2 / denom2 if denom2 != 0 else peak2
            if v2 < best_v - 1e-14:
                best_v = v2
                improved = True
            else:
                best_s[i] = -best_s[i]
        if not improved:
            break
    
    return best_s, best_v


def _refine_with_gradient(s):
    """Try a few gradient steps on a soft version to escape local minima."""
    if time.time() > _DEADLINE - _MARGIN:
        return s, _score(s)
    
    best_s = s.copy()
    best_v = _score(best_s)
    
    # Try perturbing with random noise and re-rounding
    rng = np.random.default_rng(42)
    for attempt in range(20):
        if time.time() > _DEADLINE - _MARGIN:
            break
        noise = rng.standard_normal(N) * 0.3
        x = s.astype(float) + noise
        s2 = np.sign(x)
        s2[s2 == 0] = 1.0
        v2 = _score(s2)
        if v2 < best_v:
            best_v = v2
            best_s = s2.copy()
    
    return best_s, best_v


def construct():
    t0 = time.time()
    
    # Phase 1: Differential Evolution over sine parameters
    # Parameter space: [n_sines, freqs..., amps..., phases...]
    # We'll fix n_sines to a few values and optimize freqs, amps, phases
    
    best_s = None
    best_v = np.inf
    
    # Try different numbers of sines
    for n_sines in [4, 6, 8]:
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Bounds for each parameter
        # freqs: 1 to N/2 (but we use normalized freq, so 1 to N/2)
        # amps: 0.1 to 1.0
        # phases: 0 to 2*pi
        bounds = []
        for _ in range(n_sines):
            bounds.append((1.0, N / 2.0))  # freqs
        for _ in range(n_sines):
            bounds.append((0.1, 1.0))  # amps
        for _ in range(n_sines):
            bounds.append((0.0, 2 * np.pi))  # phases
        
        def objective(params):
            if time.time() > _DEADLINE - _MARGIN:
                return 1e6
            # params: [freqs..., amps..., phases...]
            freqs = params[:n_sines]
            amps = params[n_sines:2 * n_sines]
            phases = params[2 * n_sines:3 * n_sines]
            s = _sign_vector_from_sines(freqs, amps, phases)
            return _score(s)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False,
                init='sobol'
            )
            
            if result.success or result.fun < best_v:
                freqs = result.x[:n_sines]
                amps = result.x[n_sines:2 * n_sines]
                phases = result.x[2 * n_sines:3 * n_sines]
                s = _sign_vector_from_sines(freqs, amps, phases)
                v = _score(s)
                if v < best_v:
                    best_v = v
                    best_s = s.copy()
        except Exception:
            pass
    
    # Phase 2: Try some hand-crafted constructions
    if time.time() > _DEADLINE - _MARGIN:
        if best_s is not None:
            return best_s
        return np.ones(N)
    
    # All ones
    s = np.ones(N)
    v = _score(s)
    if v < best_v:
        best_v = v
        best_s = s.copy()
    
    # Alternating
    s = np.tile([1.0, -1.0], N // 2)
    v = _score(s)
    if v < best_v:
        best_v = v
        best_s = s.copy()
    
    # Random signs with good seed
    rng = np.random.default_rng(123)
    for _ in range(10):
        if time.time() > _DEADLINE - _MARGIN:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        v = _score(s)
        if v < best_v:
            best_v = v
            best_s = s.copy()
    
    # Phase 3: Local search on best candidate
    if best_s is not None:
        best_s, best_v = _local_search(best_s, max_rounds=10)
    
    # Phase 4: Refine with perturbation
    if best_s is not None:
        best_s, best_v = _refine_with_gradient(best_s)
    
    # Phase 5: More local search after refinement
    if best_s is not None and time.time() < _DEADLINE - _MARGIN:
        best_s, best_v = _local_search(best_s, max_rounds=15)
    
    if best_s is None:
        return np.ones(N)
    
    return best_s
# EVOLVE-BLOCK-END
