"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a convex optimization approach: parameterize the power spectrum S[k] 
    (non-negative, conjugate symmetric) and minimize the peak of the linear
    autocorrelation. The linear autocorrelation is the IFFT of the zero-padded
    power spectrum. We optimize the log-power spectrum to keep it positive.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric."""
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (h_sum ** 2))

    n_fft = 2 * N  # 800 for zero-padded FFT

    # Parameterize: log of the power spectrum for non-DC bins.
    # H[k] = sqrt(S[k]) * exp(i*phi[k])
    # For real h: S[k] = S[n_fft-k], and we can set phi[k] = -phi[n_fft-k]
    # We'll optimize log(S[k]) for k in 1..n_fft/2-1, with S[0] and S[n_fft/2] fixed.
    
    def log_power_to_h(log_S):
        """Convert log power spectrum to time-domain signal h."""
        # log_S has length n_fft//2 - 1 for bins 1..n_fft//2-1
        # S[0] is DC, S[n_fft//2] is Nyquist
        S = np.zeros(n_fft)
        S[0] = 1.0  # DC component (will be adjusted)
        S[n_fft // 2] = 1.0  # Nyquist
        
        for i, ls in enumerate(log_S):
            S[i + 1] = np.exp(ls)
            S[n_fft - (i + 1)] = np.exp(ls)
        
        # Create H with zero phase (for now, we can add phase later)
        H = np.sqrt(S)
        H = H.astype(complex)
        
        # Inverse FFT to get zero-padded signal, take first N samples
        h_full = ifft(H, n=n_fft).real[:N]
        
        # Normalize so sum(h) = N
        h_sum = np.sum(h_full)
        if abs(h_sum) < 1e-10:
            return None
        h_full = h_full * (N / h_sum)
        
        return h_full

    def objective(log_S):
        """Objective: minimize the peak of the autocorrelation."""
        h = log_power_to_h(log_S)
        if h is None:
            return 1e10
        val = compute_value(h)
        return val

    # Initial guess: flat spectrum (white noise)
    log_S_init = np.zeros(n_fft // 2 - 1)
    
    # Try to optimize
    try:
        result = minimize(
            objective,
            log_S_init,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.fun < best_value:
            h_candidate = log_power_to_h(result.x)
            if h_candidate is not None:
                val = compute_value(h_candidate)
                if val < best_value:
                    best_value = val
                    best_h = h_candidate.copy()
    except Exception:
        pass

    # If time remains, try random initializations in the log-power space
    if time.time() < deadline - 20.0:
        np.random.seed(42)
        for trial in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Random log-power spectrum
            log_S_trial = np.random.randn(n_fft // 2 - 1) * 0.5
            
            h_trial = log_power_to_h(log_S_trial)
            if h_trial is None:
                continue
            
            val = compute_value(h_trial)
            if val < best_value:
                best_value = val
                best_h = h_trial.copy()

    # Final local search from best solution found
    if time.time() < deadline - 15.0:
        best_h_fft = fft(best_h, n=n_fft)
        amp = np.abs(best_h_fft)
        
        # Extract log-power for optimization
        log_S_best = np.zeros(n_fft // 2 - 1)
        for i in range(1, n_fft // 2):
            if amp[i] > 1e-10:
                log_S_best[i - 1] = np.log(amp[i] ** 2)
            else:
                log_S_best[i - 1] = -10.0
        
        try:
            result2 = minimize(
                objective,
                log_S_best,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if result2.fun < best_value:
                h_candidate = log_power_to_h(result2.x)
                if h_candidate is not None:
                    val = compute_value(h_candidate)
                    if val < best_value:
                        best_value = val
                        best_h = h_candidate.copy()
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
