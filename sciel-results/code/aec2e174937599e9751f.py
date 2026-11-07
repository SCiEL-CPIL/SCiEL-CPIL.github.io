import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # 1. Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. UCB
    # Beta should be tuned. 2.0 is standard.
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Dynamic Weighting
    # Use global uncertainty to determine exploration weight
    global_uncertainty = np.std(sigma_safe)
    
    # Map uncertainty to weight [0.2, 0.8]
    # If uncertainty is 0, weight should be low (exploit).
    # If uncertainty is high, weight should be high (explore).
    # We need a scale. Let's assume typical sigma is ~0.1 to 1.0.
    # Use a sigmoid-like mapping or linear clip.
    # Let's normalize uncertainty relative to a reference.
    # Reference: The max sigma in the current batch? Or a fixed value?
    # Using fixed value is risky.
    # Let's use the ratio of std(sigma) to max(sigma).
    # If std is close to max, uncertainty is spread out (early).
    # If std is small, uncertainty is concentrated (late).
    
    # Alternative: Just use std(sigma) directly with a soft clip.
    # w_ucb = 0.5 + 0.5 * np.tanh(global_uncertainty)
    # This maps 0 -> 0.5, high -> 1.0.
    # We want late stage to be low.
    # So w_ucb = 0.8 * np.exp(-global_uncertainty * decay) ?
    # No, high uncertainty = early = high w_ucb.
    # Low uncertainty = late = low w_ucb.
    
    # Let's use: w_ucb = 0.2 + 0.6 * (global_uncertainty / (global_uncertainty + 0.1))
    # This is a saturating function.
    # If global_uncertainty is 0.1, w = 0.2 + 0.6*0.5 = 0.5.
    # If global_uncertainty is 0.01, w = 0.2 + 0.6*0.09 = 0.25.
    # If global_uncertainty is 1.0, w = 0.2 + 0.6*0.9 = 0.74.
    
    w_ucb = 0.2 + 0.6 * (global_uncertainty / (global_uncertainty + 0.1))
    w_ei = 1.0 - w_ucb
    
    combined_score = w_ei * ei + w_ucb * ucb
    
    return combined_score
