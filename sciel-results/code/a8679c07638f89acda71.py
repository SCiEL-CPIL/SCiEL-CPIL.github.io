"""
Optimized solver for the third autocorrelation inequality task.
Uses continuous relaxation via tanh, gradient-based optimization (L-BFGS-B),
and a final discrete local search polish.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
TIME_MARGIN = 10.0  # Safety margin

# EVOLVE-BLOCK-START

def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if abs(R0) < 1e-10:
        return 1.0
    # Sidelobes are R[1:N] and R[N:2*N-1] (symmetric part)
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    C3 = max_sidelobe / R0
    return C3

def _compute_c3_grad(x):
    """
    Compute C3 and its gradient with respect to x using finite differences or analytical approximation.
    For L-BFGS-B, we can use a smooth surrogate objective function instead of the non-smooth max.
    Here we define a smooth objective: sum of squares of sidelobes / R0^2.
    This encourages reducing all sidelobes, not just the max, which often leads to a lower max.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if abs(R0) < 1e-10:
        return 1.0, np.zeros(N)
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    # Objective: minimize (sum |R[k]|^2) / R0^2
    # This is a smooth function of x.
    obj = np.sum(sidelobes**2) / (R0**2)
    
    # Gradient calculation via finite difference is too slow for N=400 in Python loop.
    # We will rely on L-BFGS-B's internal finite difference or provide a custom gradient.
    # Since providing an exact analytical gradient for this complex FFT-based objective is tricky,
    # we will use a smooth surrogate that is easier to handle or just let scipy handle finite diff.
    # However, finite diff for N=400 means 400+ FFTs per iteration, which is ~0.01s per iter.
    # 100 iters -> 1s. This is acceptable.
    
    return obj, None

def _smooth_objective(z, start_time):
    """
    Objective function for optimization in z-space.
    x = tanh(z)
    Minimize a surrogate for C3.
    """
    if time.time() - start_time > DEADLINE - TIME_MARGIN:
        return 1e10
        
    x = np.tanh(z)
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if abs(R0) < 1e-10:
        return 1e10
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    
    # Use a soft-max or sum-of-squares to make the objective smooth.
    # Sum of squares penalizes all sidelobes, leading to a more "balanced" low-sidelobe solution.
    # We normalize by R0^2 to make it scale-invariant (though R0 is fixed for tanh, it varies during optimization).
    # Actually, for tanh(z), R0 = sum(tanh(z_i)^2).
    
    # Let's minimize the sum of squared sidelobes relative to R0^2.
    # This is a proxy for minimizing the energy in the sidelobes.
    val = np.sum(sidelobes**2) / (R0**2)
    
    # Add a small penalty to encourage binarization (pushing z to +/- infinity)
    # This helps in getting closer to +/- 1 solutions.
    # Penalty: -lambda * sum(z^2) ? No, we want z to be large.
    # Actually, we just want to minimize the autocorrelation metric.
    
    return val

def _local_search_discrete(x, time_limit=None):
    """
    Discrete local search: flip single bits to reduce C3.
    """
    start_time = time.time()
    if time_limit is None:
        time_limit = 5.0
        
    R = None
    def get_R(x):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        power = F * np.conj(F)
        return np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
    R = get_R(x)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    
    # Precompute indices for fast access
    # We will iterate through random indices
    rng = np.random.RandomState(42)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(0, N)
        x[idx] = -x[idx]
        R = get_R(x)
        new_c3 = _compute_c3(x)
        
        if new_c3 < best_c3 - 1e-12:
            best_x = x.copy()
            best_c3 = new_c3
            no_improve = 0
        else:
            x[idx] = -x[idx] # revert
            no_improve += 1
            if no_improve > max_no_improve:
                break
                
    return best_x, best_c3

def _construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = None
    best_c3 = float('inf')
    
    # Strategy 1: Continuous optimization from multiple random starts
    # We use z-space where x = tanh(z).
    # We minimize a smooth surrogate of the autocorrelation sidelobe energy.
    
    num_starts = 5
    for i in range(num_starts):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
            
        # Random initialization for z
        # Start with small random z to allow gradient flow
        z0 = np.random.randn(N) * 0.1
        
        try:
            result = minimize(_smooth_objective, z0, args=(start_time,), method='L-BFGS-B',
                             options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5})
            
            x_cont = np.tanh(result.x)
            x_bin = np.sign(x_cont)
            x_bin[x_bin == 0] = 1.0
            
            c3_bin = _compute_c3(x_bin)
            
            if c3_bin < best_c3:
                best_c3 = c3_bin
                best_x = x_bin.copy()
                
        except Exception:
            continue
            
    # Strategy 2: Discrete local search from the best continuous solution
    if best_x is not None and time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        # Allocate half the remaining time to local search
        ls_time = max(1.0, remaining_time * 0.5)
        x_polished, c3_polished = _local_search_discrete(best_x, time_limit=ls_time)
        if c3_polished < best_c3:
            best_x = x_polished
            best_c3 = c3_polished

    # Strategy 3: If we still have time, try a few more random discrete starts
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining_time = DEADLINE - TIME_MARGIN - time.time()
        # Try 2 random discrete starts with short local search
        for _ in range(2):
            if time.time() > DEADLINE - TIME_MARGIN:
                break
            x_rand = np.random.choice([-1, 1], size=N)
            ls_time = max(0.5, (DEADLINE - TIME_MARGIN - time.time()) * 0.3)
            x_trial, c3_trial = _local_search_discrete(x_rand, time_limit=ls_time)
            if c3_trial < best_c3:
                best_x = x_trial
                best_c3 = c3_trial

    if best_x is None:
        # Fallback
        best_x = np.ones(N)
        
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)

# EVOLVE-BLOCK-END
