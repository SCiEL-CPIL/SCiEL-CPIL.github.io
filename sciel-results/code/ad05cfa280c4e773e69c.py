"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Knowledge Gradient-inspired approach.
    
    Combines Student-t Process EI (for heavy-tailed uncertainty modeling)
    with a structured exploration bonus that approximates the KG's
    value of information by weighting uncertainty by the probability
    of containing the global minimum.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Sanitize sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-12, posinf=1e6, neginf=1e-12)
    
    # Student-t Process EI with nu=3 for heavier tails
    nu = 3.0
    scale_factor = np.sqrt((nu - 2.0) / nu)
    t_scale = sigma_safe * scale_factor
    
    z_t = (f_best - mu) / t_scale
    z_t = np.clip(z_t, -10, 10)  # Numerical stability
    
    cdf_t = student_t.cdf(z_t, nu)
    pdf_t = student_t.pdf(z_t, nu)
    
    correction = nu / (nu - 1.0)
    ei_t = (f_best - mu) * cdf_t + t_scale * correction * pdf_t * (1.0 + z_t**2 / nu)
    ei_t = np.nan_to_num(ei_t, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Knowledge Gradient-inspired exploration bonus
    # The KG values points by how much they are expected to reduce the
    # uncertainty about the global minimum. We approximate this by
    # weighting the uncertainty by the probability that the point could
    # contain a value better than f_best.
    
    # P(f < f_best) under the t-process
    prob_improve = cdf_t
    
    # The expected reduction in the lower bound is related to:
    # E[max(0, f_best - f)] * P(f is near the minimum)
    # We use a structured bonus that is large when:
    # 1. There is high uncertainty (sigma is large)
    # 2. The probability of improvement is significant
    # 3. Early in the search (f_best is high)
    
    # Adaptive exploration weight
    # beta decays as f_best decreases, but we also modulate by
    # the local probability of improvement to focus exploration
    # where it's most likely to find the minimum
    
    # Base exploration that decays with f_best
    base_beta = 0.1 + 0.5 * (f_best / (f_best + 3.0))
    
    # KG-inspired: weight by probability of improvement
    # This focuses exploration on regions where the posterior
    # suggests the minimum might be
    kg_weight = 0.3 + 0.7 * prob_improve
    
    # Combined exploration bonus
    exploration_bonus = base_beta * kg_weight * sigma_safe
    
    # Also add a small pure uncertainty bonus to ensure
    # we don't completely ignore high-uncertainty regions
    # that have low probability of improvement (early exploration)
    pure_uncertainty_bonus = 0.05 * (f_best / (f_best + 10.0)) * sigma_safe
    
    # Final score
    scores = ei_t + exploration_bonus + pure_uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
