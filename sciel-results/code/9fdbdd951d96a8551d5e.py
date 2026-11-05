"""
Program to minimize the autocorrelation inequality value using Zadoff-Chu sequence modulation.

Strategy:
The objective function involves the maximum of the autoconvolution.
The autoconvolution is the inverse FFT of the squared magnitude of the FFT of h.
We propose a new structural family: h[n] = Re( W[n] * exp(i * (2*pi * alpha * n^2 / N + phi)) * ZC[n] ),
where ZC is a Zadoff-Chu sequence, W is a Gaussian window, alpha is a chirp rate, and phi is a phase.
This structure creates a time-varying frequency (chirp) with controlled amplitude tapering,
which is known to have low sidelobes in autocorrelation.
We optimize the parameters (window width, chirp rate, phase) using Differential Evolution,
followed by Nelder-Mead local refinement.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

def _evaluate(h):
    """Compute the objective value."""
    # h must be 1D array of length N
    s = np.sum(h)
    if s == 0:
        return 1e10
    # Full autoconvolution
    # Note: np.convolve(h, h, mode='full') has length 2N-1
    # The center is at index N-1
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def _h_from_params(params):
    """
    Reconstruct h from parameters.
    params: array of shape (3,)
    params[0] = window_width (sigma of Gaussian, relative to N)
    params[1] = chirp_rate (alpha)
    params[2] = phase (phi)
    
    We use a Zadoff-Chu sequence of length L that divides N or is close.
    Actually, Zadoff-Chu sequences have length L. We need length N=400.
    400 is not a prime. ZC sequences work best with prime lengths or powers of 2.
    Let's use a length L that is a divisor or close.
    Alternatively, we can just use the complex exponential chirp directly without ZC,
    or use a ZC of length 400? 
    Zadoff-Chu sequences are defined for lengths L where gcd(L, u)=1.
    Let's try to generate a ZC sequence of length N=400.
    u=1. gcd(400, 1)=1.
    zc[n] = exp(-i * pi * u * n * (n+1) / L) for n=0..L-1.
    
    So h[n] = Re( exp(-sigma^2 * (n - N/2)^2 / (2 * (N*params[0])^2)) * 
                  exp(i * (2 * pi * params[1] * n^2 / N + params[2])) *
                  exp(-i * pi * 1 * n * (n+1) / N) )
    
    Let's simplify the phase terms.
    Phase = 2*pi*alpha*n^2/N + phi - pi*n*(n+1)/N
           = n^2 * (2*pi*alpha/N - pi/N) - pi*n/N + phi
           = n^2 * (pi*(2*alpha - 1)/N) - pi*n/N + phi
    """
    n = np.arange(N)
    
    # Gaussian window centered at N/2
    # params[0] is relative width. Let's say sigma = params[0] * N
    sigma = params[0] * N
    if sigma < 1e-6:
        sigma = 1e-6
    window = np.exp(-(n - N/2.0)**2 / (2 * sigma**2))
    
    # Chirp + ZC phase
    alpha = params[1]
    phi = params[2]
    
    # ZC phase term: -pi * u * n * (n+1) / N, with u=1
    zc_phase = -np.pi * n * (n + 1) / N
    
    # Chirp phase term: 2 * pi * alpha * n^2 / N
    chirp_phase = 2 * np.pi * alpha * n**2 / N
    
    total_phase = chirp_phase + zc_phase + phi
    
    h = window * np.cos(total_phase)
    
    # Ensure sum is not zero and positive for stability? 
    # The objective uses sum^2, so sign doesn't matter for value, 
    # but we want to avoid sum=0.
    # If sum is close to 0, the value explodes. 
    # We can add a small DC offset if needed, but let's see if the optimizer handles it.
    # Actually, the Gaussian window is positive, and the oscillation might make sum small.
    # To be safe, we can add a small constant to ensure sum > 0.
    # But the task allows negative heights.
    # Let's just return h. The _evaluate function handles sum=0 by returning large value.
    
    return h

def _objective(params):
    """
    Wrapper for optimizer.
    params is a 1D array.
    """
    h = _h_from_params(params)
    return _evaluate(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # We will search over 3 parameters: window_width, chirp_rate, phase.
    # Bounds:
    # window_width: 0.01 to 1.0 (relative to N)
    # chirp_rate: -5 to 5
    # phase: -pi to pi
    
    bounds = [
        (0.01, 1.0),   # window_width
        (-5.0, 5.0),   # chirp_rate
        (-np.pi, np.pi) # phase
    ]
    
    try:
        # Run Differential Evolution
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=30, 
            tol=1e-4, 
            init='latinhypercube',
            seed=42,
            polish=True
        )
        
        if res.fun < best_val:
            best_h = _h_from_params(res.x)
            best_val = _evaluate(best_h)
            
    except Exception as e:
        # If DE fails, fall back to seed
        pass
    
    # If we have time left, try a local refinement from the best found so far.
    if time.time() < deadline - 10:
        try:
            # Use the best parameters from DE as initial guess for local refinement
            if 'res' in locals() and res.x is not None:
                init_params = res.x
            else:
                init_params = np.array([0.5, 0.0, 0.0])
                
            res2 = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res2.fun < best_val:
                best_h = _h_from_params(res2.x)
                best_val = _evaluate(best_h)
                
        except Exception:
            pass

    # Final check
    if _evaluate(best_h) < best_val:
        best_val = _evaluate(best_h)
        
    return best_h

# Test locally if run as main
if __name__ == "__main__":
    h = construct()
    print("Value:", _evaluate(h))
    print("Sum:", np.sum(h))
    print("Max Conv:", np.max(np.convolve(h, h)))
