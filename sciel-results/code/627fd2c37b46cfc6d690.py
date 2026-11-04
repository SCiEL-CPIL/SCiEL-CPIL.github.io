"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard: 150s wall-clock limit, keep a safety margin
_DEADLINE = 140.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _objective(params):
    """Compute the value for given chirp parameters.
    
    params: [phase, chirp_rate, amplitude_scale]
    h[i] = amplitude_scale * cos(phase + chirp_rate * i)
    
    We want to minimize: abs(2 * N * max(conv(h,h)) / sum(h)^2)
    Since sum(h) could be negative, we use sum(h)^2 which is always positive.
    The max is over signed entries of the full autoconvolution.
    """
    phase, chirp_rate, amp = params
    i = np.arange(N)
    h = amp * np.cos(phase + chirp_rate * i)
    
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e6
    
    # Full autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    
    value = abs(2.0 * N * max_val / (s * s))
    return value


def _refine_local(best_params, budget=10.0):
    """Local Nelder-Mead refinement starting from best_params."""
    if _remaining() < budget:
        return best_params
    
    # Use a few random restarts around the best point for local refinement
    best_val = _objective(best_params)
    best_p = best_params.copy()
    
    for restart in range(3):
        if _remaining() < 2.0:
            break
        # Small perturbation
        noise = np.random.uniform(-0.1, 0.1, size=3)
        x0 = best_p + noise
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x
        except Exception:
            pass
    
    return best_p


def _global_search(budget=60.0):
    """Differential evolution global search over [phase, chirp_rate, amp]."""
    if _remaining() < budget:
        return None
    
    bounds = [
        (-np.pi, np.pi),       # phase
        (-0.5, 0.5),           # chirp_rate
        (0.5, 5.0)             # amplitude scale
    ]
    
    # Determine number of generations based on remaining time
    # Each eval is fast (~ms for N=400), so we can do many
    max_iter = 50  # conservative
    
    try:
        res = differential_evolution(
            _objective, bounds,
            maxiter=max_iter,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            init='sobol',
            updating='deferred',
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        return res.x
    except Exception:
        return None


def construct():
    """Construct step heights to minimize the autocorrelation inequality value."""
    global _start_time
    _start_time = time.time()
    
    # Initial best: the seed (ones)
    best_h = np.ones(N)
    best_value = 2.0  # value for ones
    
    # Strategy: chirp representation h[i] = A * cos(phi + k*i)
    # The chirp decorrelates the autocorrelation sidelobes.
    
    # First, try a quick grid search to find a good region
    # Phase in [-pi, pi], chirp_rate in [-0.5, 0.5]
    # We'll do a coarse grid first
    
    if _remaining() > 30:
        # Coarse grid search
        phases = np.linspace(-np.pi, np.pi, 20)
        chirps = np.linspace(-0.4, 0.4, 20)
        
        for phi in phases:
            if _remaining() < 25:
                break
            for k in chirps:
                if _remaining() < 25:
                    break
                # Try a few amplitudes
                for amp in [1.0, 2.0, 3.0]:
                    i = np.arange(N)
                    h = amp * np.cos(phi + k * i)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    conv = np.convolve(h, h, mode='full')
                    val = abs(2.0 * N * np.max(conv) / (s * s))
                    if val < best_value:
                        best_value = val
                        best_h = h.copy()
    
    # Global search with differential evolution
    if _remaining() > 40:
        de_result = _global_search(budget=min(60.0, _remaining() - 15))
        if de_result is not None:
            # Evaluate the DE result
            phase, chirp_rate, amp = de_result
            i = np.arange(N)
            h_de = amp * np.cos(phase + chirp_rate * i)
            s = np.sum(h_de)
            if abs(s) > 1e-12:
                conv = np.convolve(h_de, h_de, mode='full')
                val_de = abs(2.0 * N * np.max(conv) / (s * s))
                if val_de < best_value:
                    best_value = val_de
                    best_h = h_de.copy()
    
    # Local refinement
    if _remaining() > 15:
        # Refine around the best found
        # Extract parameters from best_h if it came from a chirp
        # For simplicity, try local refinement on the chirp parameter space
        # centered at a reasonable guess
        
        # Try multiple local starts
        local_starts = []
        # From DE result if available
        if _remaining() > 20:
            # Try a few different chirp configurations
            for phi0 in [-np.pi/2, 0, np.pi/2]:
                for k0 in [-0.2, 0, 0.2]:
                    local_starts.append([phi0, k0, 2.0])
        
        for start in local_starts:
            if _remaining() < 5:
                break
            refined = _refine_local(start, budget=5.0)
            if refined is not None:
                phase, chirp_rate, amp = refined
                i = np.arange(N)
                h_ref = amp * np.cos(phase + chirp_rate * i)
                s = np.sum(h_ref)
                if abs(s) > 1e-12:
                    conv = np.convolve(h_ref, h_ref, mode='full')
                    val_ref = abs(2.0 * N * np.max(conv) / (s * s))
                    if val_ref < best_value:
                        best_value = val_ref
                        best_h = h_ref.copy()
    
    # Safety: ensure we return within deadline
    if _elapsed() > _DEADLINE - 2:
        # Return whatever we have
        pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    if abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
