"""Low-dimensional parametric chirp design for signed autocorrelation minimization."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

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


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _chirp_signal(params):
    """
    Construct a signed step function from parametric chirp parameters.
    params: [center_freq, chirp_rate, bandwidth, phase, amplitude_scale]
    """
    center_freq = params[0]
    chirp_rate = params[1]
    bandwidth = params[2]
    phase = params[3]
    
    # Frequency bins for rfft
    k = np.arange(_FFT_SIZE // 2 + 1)
    
    # Create a Gaussian-shaped magnitude spectrum centered at center_freq
    # with width bandwidth and chirp-like phase
    mag = np.exp(-((k - center_freq) ** 2) / (2 * bandwidth ** 2))
    
    # Chirp phase: quadratic in frequency
    # phase_k = phase + chirp_rate * (k - center_freq) ** 2
    ph = phase + chirp_rate * (k - center_freq) ** 2
    
    fx = mag * np.exp(1j * ph)
    
    # Ensure DC component is reasonable (non-zero sum)
    # If DC is too small, the metric explodes. 
    # We can force a minimum DC or let the optimizer handle it.
    # For chirps, DC is often small. Let's add a small DC offset to ensure validity.
    # Actually, the task is "signed step heights". 
    # If sum(x) is close to 0, the metric is ill-defined or huge.
    # Let's normalize the signal to have a specific sum, e.g., 1.0.
    # But wait, the metric is scale invariant. 
    # max|c^2 conv| / (c sum)^2 = max|conv| / sum^2.
    # So we can just normalize x to have sum=1.
    
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        # Fallback: return a simple signal if chirp results in zero sum
        x = np.ones(N)
    else:
        x = x / s  # Normalize to sum=1
    
    return x


def _parametric_objective(params):
    x = _chirp_signal(params)
    return _metric(x)


def _try_local_search_fast(x, deadline, rng, max_rounds=50000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
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
    
    rng = np.random.default_rng(2024)
    
    # Initial guess: flat signal
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Phase 1: Parametric Chirp Optimization using Differential Evolution
    # Bounds:
    # center_freq: [10, _FFT_SIZE//4] (avoiding DC and Nyquist)
    # chirp_rate: [-0.5, 0.5]
    # bandwidth: [5, 50]
    # phase: [-pi, pi]
    
    bounds = [
        (10, _FFT_SIZE // 4),
        (-0.5, 0.5),
        (5, 50),
        (-np.pi, np.pi)
    ]
    
    # Run differential evolution
    # maxiter and popsize are tuned to fit within time budget
    # DE is stochastic, so we give it a fixed seed for reproducibility if needed, 
    # but rng is used for local search.
    
    start_time = time.time()
    try:
        # We use a limited number of iterations to ensure it finishes in time.
        # 50 iterations with popsize 20 might be too slow if objective is heavy.
        # Let's check objective cost: FFT of size 512 is fast. 
        # 50 * 20 * 2 (function evals per iter approx) = 2000 evals.
        # 2000 * 0.001s = 2s. This is safe.
        
        # We can afford more. Let's try 100 iterations, popsize 30.
        # 100 * 30 * 2 = 6000 evals. ~6s.
        
        result = differential_evolution(
            _parametric_objective,
            bounds,
            maxiter=100,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,  # We will do our own polish
            update_interval=1
        )
        
        if result.fun < best_val:
            best_params = result.x
            best_x = _chirp_signal(best_params)
            best_val = _metric(best_x)
            
    except Exception as e:
        pass
    
    # Phase 2: Local Search
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 10, rng, max_rounds=50000)
    
    # Phase 3: Final Polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
