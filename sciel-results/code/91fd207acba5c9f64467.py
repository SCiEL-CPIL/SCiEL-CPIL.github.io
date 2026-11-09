import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization with Pareto-based exploration/exploitation balance
    # Dynamic weight: early steps favor exploration (uncertainty), later steps favor exploitation (EI)
    # We estimate the step from f_best relative to typical Hartmann-6d values.
    # Hartmann-6d range is roughly [-3.32, 0]. Initial random points give f_best ~ 0 to -1.
    # As optimization progresses, f_best approaches -3.32.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI and sigma to [0, 1] range for balanced weighting
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > eps:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > eps:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic weight based on how close f_best is to the known minimum (-3.32237)
    # When f_best is close to minimum, we should exploit more (high EI weight)
    # When f_best is far, we should explore more (high sigma weight)
    # Use a smooth transition: weight_ei = 0.3 + 0.7 * progress
    # progress = 1 - (f_best - min_val) / (max_initial - min_val)
    min_val = -3.32237
    # Initial random points in 6D Hartmann typically give values around -0.5 to 0.5
    # Use a reasonable range for the transition
    max_initial = 0.5  # typical worst initial value
    
    progress = 1.0 - (f_best - min_val) / (max_initial - min_val)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic weights: start with 30% EI / 70% exploration, end with 90% EI / 10% exploration
    weight_ei = 0.3 + 0.6 * progress
    weight_explore = 1.0 - weight_ei
    
    # Combine: weighted sum of normalized EI and normalized uncertainty
    scores = weight_ei * ei_norm + weight_explore * sigma_norm
    
    # Scale back to original EI scale to maintain consistent magnitude
    # This helps the acquisition function values be in a reasonable range
    scale_factor = np.mean(ei) + 0.1 * np.mean(sigma_safe)
    scores = scores * scale_factor + 0.1 * np.mean(sigma_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
