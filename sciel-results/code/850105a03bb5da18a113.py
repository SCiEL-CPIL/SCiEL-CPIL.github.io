"""
Strategy: Chirp-based spectral suppression.

The objective function involves the maximum of the autoconvolution.
In the frequency domain, the autoconvolution corresponds to the squared magnitude of the DFT.
By using a signal with a linearly varying phase (a chirp), the energy is spread out in the
frequency domain, and the time-domain autocorrelation sidelobes are suppressed due to
destructive interference.

We parameterize h as:
h[n] = A * cos(2*pi*f*n + pi*chirp*n^2) + B

This reduces the 400-dimensional optimization problem to a 3-dimensional one (A, f, chirp, B).
We use a grid search to find a good configuration, then a local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145  # wall-clock seconds; evaluator kills at 150s, keep margin


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _generate_chirp(A, f, chirp, B):
    """Generate the step heights from chirp parameters."""
    n = np.arange(N)
    phase = 2 * np.pi * f * n + np.pi * chirp * n**2
    h = A * np.cos(phase) + B
    return h


def _objective(params):
    """Objective function for optimization."""
    A, f, chirp, B = params
    h = _generate_chirp(A, f, chirp, B)
    val, valid = _evaluate(h)
    if not valid:
        return 1e10
    return val


def construct():
    start = time.time()
    
    # Fallback candidate: two-spike + small negative tail (from current best strategy)
    def _spike_candidate():
        a = 0.5
        K = 200
        eps = 0.001
        h = np.zeros(N)
        h[0] = a
        h[1] = 1.0 - a
        h[2:K] = -eps
        return h

    best_h = _spike_candidate()
    best_value, _ = _evaluate(best_h)
    
    # Grid search over chirp parameters
    # We want the sum to be non-zero, so B should be significant.
    # A controls the amplitude of the oscillation.
    # f is the starting frequency.
    # chirp is the rate of frequency change.
    
    # Coarse grid
    A_vals = [0.1, 0.5, 1.0, 2.0]
    f_vals = [0.01, 0.05, 0.1, 0.2, 0.5]
    chirp_vals = [0.001, 0.01, 0.05, 0.1, 0.5]
    B_vals = [0.1, 0.5, 1.0, 2.0]
    
    # To keep computation fast, we limit the grid size.
    # Total points: 4 * 5 * 5 * 4 = 400. Each evaluation is ~O(N^2) or O(N log N).
    # N=400, so conv is fast.
    
    for A in A_vals:
        for f in f_vals:
            for chirp in chirp_vals:
                for B in B_vals:
                    if time.time() - start > DEADLINE - 10:
                        break
                    h = _generate_chirp(A, f, chirp, B)
                    val, valid = _evaluate(h)
                    if valid and val < best_value:
                        best_value = val
                        best_h = h.copy()
    
    # Local refinement using L-BFGS-B on the 4 parameters
    if time.time() - start < DEADLINE - 5:
        # Initialize with the best found so far
        # We need to map best_h back to parameters? 
        # Actually, it's easier to just restart from a few good grid points or the current best_h 
        # by approximating its parameters or just optimizing from a few diverse starts.
        
        # Let's try optimizing from the best grid point found. 
        # But we didn't store the best params. Let's just use a few random starts in the local optimizer.
        
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() - start > DEADLINE - 5:
                break
            # Random start near the grid ranges
            A0 = rng.uniform(0.1, 2.0)
            f0 = rng.uniform(0.01, 0.5)
            c0 = rng.uniform(0.001, 0.5)
            B0 = rng.uniform(0.1, 2.0)
            x0 = np.array([A0, f0, c0, B0])
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead', # Nelder-Mead is robust for small dimensions and non-differentiable objectives
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.success:
                    h_res = _generate_chirp(*res.x)
                    val_res, valid_res = _evaluate(h_res)
                    if valid_res and val_res < best_value:
                        best_value = val_res
                        best_h = h_res.copy()
            except Exception:
                pass

    # Final validity guard
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _spike_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return best_h
