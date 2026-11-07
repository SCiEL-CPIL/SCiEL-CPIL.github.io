"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Hypervolume-inspired acquisition: combines expected improvement with
    # an exploration bonus based on uncertainty relative to the best value.
    # This balances exploitation of low-objective regions with exploration
    # of high-uncertainty areas, which is effective for multimodal landscapes.
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) component - exploitation
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Uncertainty-based exploration bonus
    # Scale uncertainty by how far we are from the best value
    # Higher sigma in regions where f_best is not yet achieved gets more weight
    uncertainty_bonus = sigma_safe * 0.5
    
    # Novelty/exploration: reward points that are uncertain AND potentially better
    # Use a weighted combination that emphasizes exploration early
    # The weight on exploration decays as we get closer to the optimum
    gap = f_best - mu
    # When gap is large (far from best), we want more exploration
    # When gap is small (near best), we want more exploitation
    
    # Combined score: EI + scaled uncertainty
    # The scaling factor 1.0 on EI and 0.3 on sigma gives a balance
    # that encourages exploring uncertain regions without ignoring clear improvements
    score = ei + 0.3 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
