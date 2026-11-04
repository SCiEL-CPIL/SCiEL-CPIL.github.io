"""
Signed autocorrelation maximum absolute-value variant.
Strategy: Phase-only spectral design with smooth-max surrogate optimization.
Fix the power spectrum to be flat (constant magnitude) and optimize the complex phase
vector to minimize the maximum absolute off-diagonal autoconvolution.
Use stochastic gradient descent with a smooth max surrogate for the objective.
"""
N = 400
DEADLINE = 145  # safety margin before 150s limit

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _autoconv(x):
    """Compute linear autoconvolution."""
    return fftconvolve(x, x, mode='full')

def _metric(x):
    """
    Compute max |autoconv| / (sum x)^2, excluding the DC term (lag 0).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    c = _autoconv(x)
    # Lag 0 is at index N-1
    c_side = np.concatenate([c[:N-1], c[N:]])
    max_side = np.max(np.abs(c_side))
    return max_side / (s * s)

def _smooth_max_abs(c_side, beta=10.0):
    """
    Smooth approximation of max(|c_side|) using log-sum-exp.
    c_side: array of values
    beta: temperature parameter (higher = sharper)
    """
    abs_c = np.abs(c_side)
    # Shift for numerical stability
    m = np.max(abs_c)
    if m < 1e-10:
        return m
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _phase_to_signal(phase):
    """
    Convert a phase vector to a real signal with flat power spectrum.
    phase: array of length N (phases in radians)
    Returns: real signal x of length N
    """
    # Flat magnitude spectrum
    magnitude = np.ones(N)
    # Complex spectrum
    spectrum = magnitude * np.exp(1j * phase)
    # Inverse FFT to get time-domain signal
    x = np.fft.ifft(spectrum).real
    # Ensure the signal is real and has zero mean offset issues handled by the metric
    return x

def _objective_phase(phase, beta):
    """
    Objective function for phase optimization.
    Returns smooth max absolute autoconvolution / (sum x)^2
    """
    x = _phase_to_signal(phase)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv(x)
    c_side = np.concatenate([c[:N-1], c[N:]])
    smooth_max = _smooth_max_abs(c_side, beta)
    return smooth_max / (s * s)

def _gradient_descent_phase(phase, beta, learning_rate, max_iter, deadline):
    """
    Stochastic gradient descent on the phase vector.
    """
    t0 = time.time()
    current_phase = phase.copy()
    current_obj = _objective_phase(current_phase, beta)
    
    # Precompute some constants
    n = N
    
    for it in range(max_iter):
        if time.time() - t0 > deadline - 2:
            break
        
        # Compute gradient numerically (finite differences)
        # For efficiency, we could use analytic gradients, but let's start with FD
        # Actually, let's use a simple approach: perturb each phase element
        
        # Use a small perturbation
        eps = 1e-3
        grad = np.zeros(n)
        
        # For speed, only compute gradient for a subset of elements each iteration
        # Or compute full gradient but with fewer iterations
        # Let's compute full gradient
        for i in range(n):
            phase_plus = current_phase.copy()
            phase_plus[i] += eps
            obj_plus = _objective_phase(phase_plus, beta)
            
            phase_minus = current_phase.copy()
            phase_minus[i] -= eps
            obj_minus = _objective_phase(phase_minus, beta)
            
            grad[i] = (obj_plus - obj_minus) / (2 * eps)
        
        # Gradient descent step
        current_phase -= learning_rate * grad
        
        # Wrap phases to [-pi, pi]
        current_phase = (current_phase + np.pi) % (2 * np.pi) - np.pi
        
        # Update objective
        new_obj = _objective_phase(current_phase, beta)
        
        if new_obj < current_obj:
            current_obj = new_obj
        else:
            # Revert if it got worse
            current_phase = (current_phase + learning_rate * grad)
            current_phase = (current_phase + np.pi) % (2 * np.pi) - np.pi
    
    return current_phase, current_obj

def _local_search_heights(x, deadline):
    """
    Local search: flip signs of individual elements to improve metric.
    """
    t0 = time.time()
    current_val = _metric(x)
    n = len(x)
    x = x.copy()
    
    improved = True
    while improved and (time.time() - t0 < deadline - 5):
        improved = False
        for i in range(n):
            if time.time() - t0 > deadline - 5:
                break
            
            old_val = x[i]
            x[i] = -old_val
            
            new_val = _metric(x)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                x[i] = old_val
    
    return x, current_val

def construct():
    """
    Returns signed step heights.
    Uses phase-only spectral design with smooth-max surrogate optimization,
    followed by local sign-flip refinement.
    """
    t0 = time.time()
    
    # Best result so far
    best_x = np.ones(N, dtype=np.float64)
    best_val = _metric(best_x)
    
    # Try different beta values for the smooth max
    for beta in [5.0, 10.0, 20.0]:
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # Initialize phase randomly
        np.random.seed(42 + int(beta))
        phase = np.random.uniform(-np.pi, np.pi, N)
        
        # Run gradient descent
        learning_rate = 0.1
        max_iter = 200
        phase_opt, obj_opt = _gradient_descent_phase(phase, beta, learning_rate, max_iter, DEADLINE - 10)
        
        # Convert to signal
        x_candidate = _phase_to_signal(phase_opt)
        
        # The signal from phase optimization may have arbitrary scale
        # Normalize to have unit sum (or keep as is)
        s = np.sum(x_candidate)
        if abs(s) > 1e-10:
            # Scale to have sum = 1 for consistent comparison
            x_candidate = x_candidate / s
        
        val_candidate = _metric(x_candidate)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_x = x_candidate.copy()
        
        # Also try with the raw (unnormalized) signal
        x_raw = _phase_to_signal(phase_opt)
        val_raw = _metric(x_raw)
        if val_raw < best_val:
            best_val = val_raw
            best_x = x_raw.copy()
    
    # Local refinement: flip signs
    if time.time() - t0 < DEADLINE - 15:
        x_refined, val_refined = _local_search_heights(best_x, DEADLINE)
        if val_refined < best_val:
            best_x = x_refined
            best_val = val_refined
    
    return best_x

# EVOLVE-BLOCK-END
