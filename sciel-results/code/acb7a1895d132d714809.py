"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct a signed step function (heights array) that minimizes the maximum
    absolute autoconvolution divided by the squared integral.
    
    We use a random initialization with sign structure and then optimize using
    gradient-free methods to reduce the max |autocorr| / (sum)^2.
    
    The objective is: max_k |sum_i f[i] f[i+k]| / (sum f)^2
    where k ranges over valid lags.
    
    We normalize so that sum(f) = 1 to simplify, then minimize max |autocorr|.
    """
    np.random.seed(42)
    
    def compute_objective(f):
        """Compute max |autocorr| / (sum f)^2 for a given signal f."""
        s = np.sum(f)
        if abs(s) < 1e-12:
            return 1e10
        # Autoconvolution (linear)
        ac = np.correlate(f, f, mode='full')
        # The full autoconvolution has length 2*N-1
        # We want the maximum absolute value of the autoconvolution
        max_abs_ac = np.max(np.abs(ac))
        return max_abs_ac / (s * s)
    
    def gradient_estimate(f, eps=1e-4):
        """Numerical gradient estimation."""
        grad = np.zeros_like(f)
        base = compute_objective(f)
        for i in range(len(f)):
            f_plus = f.copy()
            f_plus[i] += eps
            f_minus = f.copy()
            f_minus[i] -= eps
            fp = compute_objective(f_plus)
            fm = compute_objective(f_minus)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Start with a random sign pattern
    f0 = np.random.choice([-1, 1], size=N)
    # Normalize
    f0 = f0 / np.sum(f0)
    
    # Use Nelder-Mead for optimization (derivative-free)
    # This can be slow, so let's try a few iterations
    
    # Actually, let's try a smarter approach: use a structured pattern
    # that naturally has low auto-correlation peaks.
    
    # Approach: Use a random binary signal, then apply iterative sign flipping
    # to reduce the peak auto-correlation.
    
    f = np.random.choice([-1, 1], size=N).astype(float)
    
    # Iterative sign flipping (greedy)
    for iteration in range(200):
        # Compute autoconvolution
        ac = np.correlate(f, f, mode='full')
        s = np.sum(f)
        
        # Find the index that contributes most to the peak
        # The peak is at some lag k, and we want to reduce |ac[k]|
        # Flipping sign at position i changes ac by -2*f[i]*f[i+k] for lag k
        # or -2*f[i]*f[i-k] depending on direction
        
        # Find the lag with maximum absolute value
        peak_lag = np.argmax(np.abs(ac))
        # Convert from full mode index to actual lag
        # In 'full' mode, index 0 corresponds to lag -(N-1), index N-1 corresponds to lag 0
        lag = peak_lag - (N - 1)
        
        # Find the position i that, if flipped, most reduces the peak
        # Flipping f[i] changes ac[lag] by -2 * f[i] * f[i + lag] (if i+lag is in range)
        # We want to minimize |ac[lag] - 2*f[i]*f[i+lag]|
        
        best_i = -1
        best_val = np.inf
        
        # Only consider flipping positions that affect the peak lag
        if lag >= 0:
            # ac[lag] = sum_i f[i] * f[i+lag]
            # Flipping f[i] changes it by -2*f[i]*f[i+lag]
            for i in range(N - lag):
                change = -2 * f[i] * f[i + lag]
                new_val = abs(ac[peak_lag] + change)
                if new_val < best_val:
                    best_val = new_val
                    best_i = i
        else:
            # ac[lag] = sum_i f[i] * f[i - |lag|]
            lag_abs = -lag
            for i in range(lag_abs, N):
                change = -2 * f[i] * f[i - lag_abs]
                new_val = abs(ac[peak_lag] + change)
                if new_val < best_val:
                    best_val = new_val
                    best_i = i
        
        if best_i >= 0:
            f[best_i] *= -1
    
    # Normalize
    s = np.sum(f)
    if abs(s) > 1e-12:
        f = f / s
    
    return f
# EVOLVE-BLOCK-END
