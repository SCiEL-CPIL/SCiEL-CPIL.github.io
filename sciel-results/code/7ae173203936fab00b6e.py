# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function.
    
    Approximates the expected improvement in the global minimum by
    using a Monte Carlo estimate of the posterior predictive distribution
    at the top-k candidates.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means more promising
    """
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Select top-k candidates based on a simple heuristic (low mu + high sigma)
    # This helps focus the KG computation on promising regions
    # Use a combination of mean and uncertainty to select candidates
    k = min(50, n)
    
    # Score candidates for selection: we want low mu and high sigma
    # Use a simple UCB-like score for selection
    selection_score = mu - 1.0 * sigma_safe
    top_k_indices = np.argsort(selection_score)[:k]
    
    # For each candidate, estimate the KG
    # KG(i) = E[max(0, f_best - min(f_i, f_best))]
    # where f_i is the value at candidate i, and we consider the joint distribution
    # with the other top-k candidates
    
    # We'll use Monte Carlo to estimate the KG
    n_mc = 100  # Number of Monte Carlo samples
    
    # Pre-compute some values
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    # For efficiency, we'll compute KG for all n candidates, but use the top-k
    # to determine the "current best" in the MC samples
    
    # Initialize scores
    scores = np.zeros(n)
    
    # For each candidate i, KG(i) = E[max(0, f_best - min(f_i, f_best))]
    # = E[max(0, f_best - min(f_i, f_best))]
    # where f_i ~ N(mu[i], sigma[i]^2)
    
    # We can compute this analytically for the simple case where we only
    # consider candidate i vs f_best:
    # KG_simple(i) = E[max(0, f_best - min(f_i, f_best))]
    # = E[max(0, f_best - f_i)] * P(f_i < f_best) + 0 * P(f_i >= f_best)
    # Wait, that's not quite right. Let me think again.
    
    # min(f_i, f_best) = f_i if f_i < f_best, else f_best
    # max(0, f_best - min(f_i, f_best)) = 
    #   max(0, f_best - f_i) if f_i < f_best
    #   max(0, f_best - f_best) = 0 if f_i >= f_best
    # So KG_simple(i) = E[max(0, f_best - f_i) * I(f_i < f_best)]
    # = E[max(0, f_best - f_i)] since if f_i >= f_best, max(0, f_best - f_i) = 0
    # This is just EI!
    
    # But the full KG considers that evaluating at i might reveal that another
    # point j (from the top-k) is actually better. So:
    # KG(i) = E[max(0, f_best - min(f_i, f_j1, f_j2, ...))]
    # where the min is over all top-k candidates including i
    
    # For a more accurate KG, we need to consider the joint distribution.
    # Since we don't have the full covariance matrix, we'll approximate by
    # assuming independence between the top-k candidates (which is a common
    # approximation when the full kernel matrix isn't available).
    
    # With independence assumption:
    # KG(i) = E[max(0, f_best - min(f_i, f_j for j in top-k, j != i))]
    # We can estimate this with MC:
    # 1. Sample f_i ~ N(mu[i], sigma[i]^2)
    # 2. Sample f_j ~ N(mu[j], sigma[j]^2) for j in top-k, j != i
    # 3. Compute min(f_i, f_j for j in top-k, j != i)
    # 4. Compute max(0, f_best - min_value)
    # 5. Average over MC samples
    
    # However, doing this for all n candidates with MC is expensive.
    # Let's use a hybrid approach:
    # - For the top-k candidates, use MC-based KG
    # - For the rest, use a simpler approximation (EI or UCB)
    
    # First, compute EI for all candidates as a baseline
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For the top-k candidates, compute a more accurate KG using MC
    # We'll assume independence for the MC samples
    
    # Pre-generate MC samples for the top-k candidates
    # For each of the n_mc samples, draw from each top-k candidate
    rng = np.random.default_rng(42)
    
    # Sample from each top-k candidate
    # Shape: (n_mc, k)
    mc_samples = rng.normal(loc=mu_top, scale=sigma_top, size=(n_mc, k))
    
    # For each top-k candidate i, compute KG
    for idx in range(k):
        # Get the sample for candidate i
        f_i = mc_samples[:, idx]  # shape (n_mc,)
        
        # Get the min of all other top-k candidates
        # Create a copy and set idx to inf
        other_samples = mc_samples.copy()
        other_samples[:, idx] = np.inf
        min_other = np.min(other_samples, axis=1)  # shape (n_mc,)
        
        # min(f_i, min_other)
        min_val = np.minimum(f_i, min_other)
        
        # max(0, f_best - min_val)
        kg_val = np.maximum(0, f_best - min_val)
        
        # Average over MC samples
        scores[top_k_indices[idx]] = np.mean(kg_val)
    
    # For candidates not in top-k, use EI as a reasonable proxy
    # Scale EI to be comparable to KG values
    # The KG for a candidate that is already in top-k should be >= EI
    # For others, EI is a lower bound on KG
    mask_not_topk = np.ones(n, dtype=bool)
    mask_not_topk[top_k_indices] = False
    
    # Use EI for non-top-k candidates, but scale it slightly
    # to account for the fact that KG >= EI
    scores[mask_not_topk] = ei[mask_not_topk]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
