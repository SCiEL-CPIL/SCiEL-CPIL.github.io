# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using
    Bayesian Regret Minimization (BRM) principles.
    
    The score is computed as the Expected Improvement (EI) weighted by the
    probability of the candidate being the global minimum among all candidates
    (approximated via pairwise comparisons), normalized by the evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    N = len(mu)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Probability of being the ArgMin (BRM component)
    # Approximation: P(y_i = min_j y_j) = prod_{j != i} P(y_i < y_j)
    # Assuming independence of the posterior samples for the rank calculation.
    
    # Vectorized computation
    # mu_diff[i, j] = mu_j - mu_i
    mu_diff = mu[None, :] - mu[:, None]
    
    # sigma_sum_sq[i, j] = sigma_i^2 + sigma_j^2
    sigma_sum_sq = sigma[:, None]**2 + sigma[None, :]**2
    
    # Z[i, j] = (mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2)
    # Note: P(y_i < y_j) = Phi( (mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2) )
    Z = mu_diff / np.sqrt(sigma_sum_sq)
    
    # Compute CDF
    P_less = norm.cdf(Z)
    
    # Exclude self-comparison (i == j)
    # P(y_i < y_i) is 0.5, but we want to exclude it from the product.
    # Set diagonal to 1.0 so that log(1.0) = 0.
    np.fill_diagonal(P_less, 1.0)
    
    # Clip to avoid log(0)
    P_less = np.clip(P_less, 1e-12, 1.0)
    
    # Log-probability of being the argmin
    log_P_argmin = np.sum(np.log(P_less), axis=1)
    
    # Normalize log-probabilities to get relative probabilities in [0, 1]
    # This prevents underflow and ensures the best candidate has a relative prob of 1.
    max_log = np.max(log_P_argmin)
    normalized_log_P = log_P_argmin - max_log
    prob_argmin = np.exp(normalized_log_P)
    
    # 3. Combine EI and ProbArgMin, weighted by cost
    # Score = (EI * ProbArgMin) / Cost
    scores = (ei * prob_argmin) / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
