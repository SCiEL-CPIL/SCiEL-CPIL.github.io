"""Parametric chirp design for signed autocorrelation minimization."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _chirp_signal(params):
    """
    Construct a signed step function from parametric chirp parameters.
    params: [chirp_rate, center_freq, phase, bandwidth]
    
    The signal is modeled as the real part of a complex Zadoff-Chu like sequence:
    x[n] = A * cos(2*pi * (center_freq * n + 0.5 * chirp_rate * n^2) + phase) * window[n]
    
    We use a simple window (e.g., rectangular or slightly tapered) to ensure finite energy and avoid edge effects,
    although the metric is scale invariant and translation invariant (mostly).
    Actually, for N=400, a rectangular window on the chirp is a standard CAZAC (Constant Amplitude Zero Autocorrelation) sequence.
    The autocorrelation of a complex CAZAC sequence is an impulse (zero off-peak).
    For the real part, the autocorrelation will have side lobes, but they should be significantly lower than random signals.
    """
    chirp_rate = params[0]
    center_freq = params[1]
    phase = params[2]
    bandwidth = params[3]  # Not strictly used in pure CAZAC, but can modulate amplitude if we want to shape spectrum
    
    n = np.arange(N)
    
    # Quadratic phase term
    phase_term = 2 * np.pi * (center_freq * n + 0.5 * chirp_rate * n**2) + phase
    
    # Real part
    x = np.cos(phase_term)
    
    # Optional: Apply a mild window to reduce spectral leakage effects if needed, 
    # but CAZAC properties are best with rectangular. 
    # Let's stick to rectangular for now to preserve CAZAC property as much as possible.
    
    return x


def _parametric_objective(params):
    x = _chirp_signal(params)
    return _metric(x)


def _try_chirp_optimization(deadline, rng):
    """
    Optimize the 4 chirp parameters using Differential Evolution.
    """
    # Bounds for parameters:
    # chirp_rate: 0 to 0.5 (Nyquist limit for quadratic term slope)
    # center_freq: -0.5 to 0.5
    # phase: 0 to 2*pi
    # bandwidth: 0.5 to 1.0 (scaling factor, though metric is scale invariant, 
    #             but if we introduce windowing later, it might matter. 
    #             For now, fix to 1.0 or optimize it slightly if it helps with numerical stability?)
    # Actually, since metric is scale invariant, amplitude doesn't matter.
    # So we only need 3 parameters: chirp_rate, center_freq, phase.
    
    bounds = [
        (0.0, 0.5),      # chirp_rate
        (-0.5, 0.5),     # center_freq
        (0.0, 2 * np.pi) # phase
    ]
    
    # Initial best
    best_x = _chirp_signal([0.1, 0.0, 0.0])
    best_val = _metric(best_x)
    
    # Run Differential Evolution
    # maxiter needs to be large enough, but we have time constraints.
    # popsize=20 is standard.
    # tol=1e-6
    
    def objective(params):
        return _parametric_objective(params)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=30,
            tol=1e-5,
            seed=42,
            polish=True,  # Use L-BFGS-B for final polish
            x0=None
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = _chirp_signal(result.x)
            
    except Exception:
        pass
    
    # Additional local refinement using Nelder-Mead on the 3 params
    if time.time() < deadline - 10:
        x0 = [0.1, 0.0, 0.0]
        # Try a few random starts for NM
        for _ in range(5):
            x0 = [
                rng.uniform(0, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0, 2 * np.pi)
            ]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = _chirp_signal(res.x)
            except Exception:
                pass
            
            if time.time() > deadline - 5:
                break
                
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=50000):
    """Fast local search with in-place modifications on the time-domain signal."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.02, 0.02
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    # Initialize with a random chirp
    rng = np.random.default_rng(42)
    init_params = [0.1, 0.0, 0.0]
    best_x = _chirp_signal(init_params)
    best_val = _metric(best_x)
    
    # Phase 1: Parametric Chirp Optimization (Core Strategy)
    if time.time() < deadline - 30:
        x_chirp, val_chirp = _try_chirp_optimization(deadline - 30, rng)
        if val_chirp < best_val:
            best_val = val_chirp
            best_x = x_chirp
    
    # Phase 2: Local Search on Time Domain
    # This allows the signal to deviate from the strict parametric form if it helps
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=20000)
    
    # Phase 3: Final Polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
