import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

N = 50

def construct():
    """Construct 50 nonnegative step heights to maximize the C2 lower bound."""
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # The score is ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # This is scale-invariant, so we can fix ||h||_1 = 1 and maximize ||h||_2^2 / ||h*h||_inf
    
    # We'll use a log-space optimization to handle the max operator via log-sum-exp
    # Let h = exp(x) / sum(exp(x)) to enforce sum(h) = 1 and h > 0
    # Then maximize log(||h||_2^2) - log(||h*h||_inf)
    
    # Autoconvolution: (h*h)[k] = sum_i h[i] * h[k-i] (with appropriate indexing)
    # For discrete heights on a grid, the autoconvolution at lag 0 is sum h[i]^2
    # The supremum norm is max over all lags
    
    def objective(x):
        """
        x: log-heights (unnormalized)
        Returns negative of the log-score for minimization
        """
        # Normalize to get h with sum = 1
        # Use stable log-sum-exp for normalization
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        h = exp_x / np.sum(exp_x)
        
        # Compute autoconvolution
        # The autoconvolution of a sequence h of length N on equal intervals
        # The peaks are at various lags. For nonnegative h, the max is at lag 0
        # Actually, for nonnegative sequences, the autoconvolution at lag 0 is sum(h_i^2)
        # and at other lags it's sum(h_i * h_{i+lag})
        # The supremum norm is the maximum over all lags
        
        # Compute all autoconvolution values
        # Using FFT for efficiency
        h_pad = np.zeros(2 * N - 1)
        h_pad[:N] = h
        conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
        
        # The autoconvolution values are conv[0:2*N-1]
        # For nonnegative h, the max should be at lag 0 (conv[0] = sum h_i^2)
        # But let's take the max to be safe
        conv_peak = np.max(conv[:2*N-1])
        
        # log(||h||_2^2) = log(sum h_i^2) = log(conv[0])
        log_l2_sq = np.log(conv[0] + 1e-300)
        
        # log(||h*h||_inf) = log(conv_peak)
        log_conv_peak = np.log(conv_peak + 1e-300)
        
        # We want to maximize log_l2_sq - log_conv_peak
        # So minimize the negative
        return -(log_l2_sq - log_conv_peak)
    
    # Try multiple random initializations
    best_score = 0.0
    best_h = None
    
    n_trials = 10
    
    for trial in range(n_trials):
        if time.time() - start_time > deadline:
            break
        
        # Random initialization
        if trial == 0:
            # Start with uniform
            x0 = np.zeros(N)
        elif trial == 1:
            # Start with concentrated in the middle
            x0 = np.linspace(-2, 2, N)
        elif trial == 2:
            # Start with a Gaussian-like shape
            centers = np.linspace(-1, 1, N)
            x0 = -centers ** 2
        elif trial == 3:
            # Start with alternating high/low
            x0 = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(N)])
        elif trial == 4:
            # Start with edge-concentrated
            x0 = np.linspace(2, -2, N)
        else:
            # Random
            x0 = np.random.randn(N) * 0.5
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            
            if res.success or res.fun < 10:  # accept if reasonable
                x_opt = res.x
                x_max = np.max(x_opt)
                exp_x = np.exp(x_opt - x_max)
                h_opt = exp_x / np.sum(exp_x)
                
                # Compute the actual score
                h_pad = np.zeros(2 * N - 1)
                h_pad[:N] = h_opt
                conv = np.fft.irfft(np.fft.rfft(h_pad) ** 2)
                l2_sq = np.sum(h_opt ** 2)
                l1 = np.sum(h_opt)
                conv_inf = np.max(conv[:2*N-1])
                
                if l1 > 0 and conv_inf > 0:
                    score = l2_sq / (l1 * conv_inf)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
                    
        except Exception:
            continue
    
    # If no valid solution found, return uniform
    if best_h is None:
        return [1.0] * N
    
    # Scale doesn't matter for the ratio, but return the heights
    # The score is computed from the ratio, so any scaling works
    # Return the normalized version (sum = 1)
    return list(best_h)
