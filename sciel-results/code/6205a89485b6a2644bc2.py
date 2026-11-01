"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()


def _c3_value(h):
    """Compute C3 bound: sum of |sidelobe|^3 normalized."""
    F = np.fft.rfft(h, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)


def _smooth_c3(x, beta=20.0):
    """Smooth surrogate for C3 using soft-abs on continuous variables."""
    # x is continuous in [-1, 1], length N
    F = np.fft.rfft(x, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    # Soft approximation of |s|^3
    abs_s = np.sqrt(sidelobes ** 2 + 1.0 / beta)
    cubes = abs_s ** 3
    return np.sum(cubes) / (N ** 3)


def _grad_smooth_c3(x, beta=20.0):
    """Gradient of smooth C3 with respect to x."""
    # Use numerical gradient via FFT for efficiency
    # d/dx_i of sum_j |acf_j|^3
    # This is complex; use finite differences with small steps for accuracy
    # Actually, let's use the analytical approach:
    # acf = ifft(|fft(x)|^2)
    # d(ACF)/dx is related to convolution with x
    
    # For efficiency, use central difference with small step
    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_smooth_c3(x_plus, beta) - _smooth_c3(x_minus, beta)) / (2 * eps)
    return grad


def _local_search_polish(h, deadline, max_rounds=50):
    """Deterministic local search: flip single bits to reduce C3."""
    best_h = h.copy()
    best_val = _c3_value(best_h)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            best_h[i] = -best_h[i]
            val = _c3_value(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = -best_h[i]  # revert
        if not improved:
            break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous relaxation via L-BFGS-B
    # Try multiple random initializations in continuous space
    n_continuous_starts = 3
    per_start_time = (deadline - start_time - 30.0) / n_continuous_starts
    
    for s in range(n_continuous_starts):
        phase_deadline = start_time + 30.0 + per_start_time * (s + 1)
        if time.time() >= phase_deadline - 5.0:
            break
        
        # Initialize with random values in [-1, 1]
        x_init = rng.uniform(-1.0, 1.0, size=N)
        
        # Run L-BFGS-B
        try:
            result = opt.minimize(
                _smooth_c3,
                x_init,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() >= phase_deadline - 5.0:
                break
            
            # Discretize: round to nearest ±1
            h_disc = np.sign(result.x)
            h_disc[h_disc == 0] = 1.0  # handle exact zeros
            
            val = _c3_value(h_disc)
            if val < best_val:
                best_h = h_disc.copy()
                best_val = val
                
        except Exception:
            continue
    
    # Phase 2: Also try a few discrete random starts with SA for comparison
    n_discrete_starts = 2
    per_discrete_time = (deadline - start_time - 30.0) / n_discrete_starts
    
    for s in range(n_discrete_starts):
        phase_deadline = start_time + 30.0 + per_discrete_time * (s + 1)
        if time.time() >= phase_deadline - 5.0:
            break
        
        h_init = rng.choice([-1.0, 1.0], size=N)
        h_init_val = _c3_value(h_init)
        
        if h_init_val < best_val:
            best_h = h_init.copy()
            best_val = h_init_val
    
    # Phase 3: Local search polish on the best result
    if best_h is not None and time.time() < deadline:
        polished_h, polished_val = _local_search_polish(best_h, deadline)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    if best_h is None:
        # Fallback: return random
        best_h = rng.choice([-1.0, 1.0], size=N)
    
    return best_h
