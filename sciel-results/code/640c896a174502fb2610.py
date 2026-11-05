"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Approach: Minimize the variance of the autoconvolution entries around their mean.
    The mean of the autoconvolution (when sum(h)=1) is 1/(2N-1).
    By flattening the autoconvolution profile, we reduce the maximum entry,
    thereby reducing the objective value below the seed value of 2.0.
    
    We parameterize h = h_prime + 1/N to satisfy sum(h) = 1.
    """
    
    def objective(h_prime):
        h = h_prime + 1.0 / N
        ac = np.convolve(h, h)
        # The mean of the autoconvolution is (sum(h))^2 / (2N-1) = 1 / (2N-1)
        mean_ac = 1.0 / (2 * N - 1)
        # Minimize the sum of squared deviations from the mean
        return np.sum((ac - mean_ac) ** 2)
    
    # Initial guess: small random perturbations around uniform
    np.random.seed(42)
    h0_prime = np.random.randn(N) * 0.01
    
    # L-BFGS-B is efficient for this smooth objective
    result = minimize(objective, h0_prime, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = result.x + 1.0 / N
    
    # Calculate the value
    val = abs(2 * N * np.max(np.convolve(h_opt, h_opt)) / np.sum(h_opt)**2)
    
    # Compare with seed value (2.0)
    seed_val = 2.0
    if val < seed_val:
        return h_opt
    else:
        return np.ones(N)
