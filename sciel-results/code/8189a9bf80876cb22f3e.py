"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        KG scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_candidates = mu.shape[0]
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-scores: how many standard deviations below f_best each candidate's mean is
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI) for each candidate
    # EI_i = (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i)
    # This is the expected reduction in the minimum if we evaluate candidate i
    # and it turns out to be better than f_best.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # The Knowledge Gradient is the expected reduction in the minimum observed value.
    # For a single evaluation, the KG is exactly the Expected Improvement.
    # However, to better balance exploration and exploitation in multi-modal functions,
    # we can use a modified version that accounts for the probability of each candidate
    # being the new best.
    
    # A more sophisticated KG approximation:
    # The true KG considers the joint distribution of all candidates, but that's
    # computationally expensive. A common approximation is to use EI directly,
    # which is what we've computed above.
    
    # Another approach: use a weighted combination that emphasizes candidates
    # with high EI but also some exploration for candidates with high uncertainty.
    
    # Let's use a variant that scales EI by a factor that increases with uncertainty,
    # to encourage exploration in regions with high sigma.
    
    # Actually, the standard EI already balances exploration and exploitation:
    # - When mu is close to f_best and sigma is small, EI is small.
    # - When mu is much less than f_best (good prediction), EI is large.
    # - When sigma is large, EI gets a boost from the sigma * pdf(z) term.
    
    # For Hartmann 3D, which is multi-modal, we want to ensure we explore
    # different basins. The EI naturally does this because regions with high
    # uncertainty (high sigma) get higher EI scores.
    
    # Let's return the EI scores directly.
    return ei
# EVOLVE-BLOCK-END
