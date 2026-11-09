import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return pure PMM scores
        scores = -np.asarray(mu, dtype=np.float64)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to prevent division by zero and numerical instability
    min_sigma = 1e-6
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > min_sigma), sigma, min_sigma)
    
    # Compute z-scores for potential improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # PES-inspired score: rewards high uncertainty with non-negligible probability of new best
    # sigma * (1 - Phi(z)) = sigma * P(f(x) < f_best)
    pes_scores = sigma_safe * (1.0 - Phi_z)
    
    # Adaptive weighting based on how close f_best is to the minimum (0)
    # When f_best is close to 0, favor exploitation (EI and -mu)
    # When f_best is far from 0, favor exploration (PES)
    
    # Use a sigmoid-like function to get a weight in [0, 1]
    # As f_best -> 0, weight -> 1 (more exploitation)
    # As f_best -> large, weight -> 0 (more exploration)
    scale = 0.3
    exploit_weight = 1.0 / (1.0 + np.exp(scale * f_best))
    
    # Normalize scores to be comparable in magnitude
    # Compute robust scales for each score component
    ei_scale = np.std(ei_scores) + 1e-10
    pes_scale = np.std(pes_scores) + 1e-10
    mu_scale = np.std(mu) + 1e-10
    
    # Normalize each score to have similar magnitude
    ei_norm = ei_scores / ei_scale
    pes_norm = pes_scores / pes_scale
    mu_norm = -mu / mu_scale
    
    # Blend: when exploiting, use EI + PMM; when exploring, use PES
    # exploit_score = combination of EI and PMM
    # explore_score = PES
    exploit_score = 0.6 * ei_norm + 0.4 * mu_norm
    explore_score = pes_norm
    
    # Final score: weighted combination
    scores = exploit_weight * exploit_score + (1.0 - exploit_weight) * explore_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
