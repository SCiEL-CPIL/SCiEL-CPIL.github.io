import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    n = len(mu)
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Calculate probability that each point is better than f_best
    # P(f_i < f_best)
    z_best = (f_best - mu) / (sigma + 1e-9)
    p_beat_best = norm.cdf(z_best)
    
    # Calculate probability that each point is the minimum among all candidates
    # Assuming independence between candidates
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    
    # We need to compute this for all pairs (i, j) where i != j
    # Vectorized approach:
    # mu_diff[i, j] = mu[j] - mu[i]
    # sigma_diff[i, j] = sqrt(sigma[i]^2 + sigma[j]^2)
    
    # To save memory and time, we can do it in chunks or just full matrix if N=2000 is small enough
    # 2000x2000 is 4M elements. 4M * 8 bytes = 32MB. This is fine.
    
    mu_col = mu[:, np.newaxis] # (N, 1)
    mu_row = mu[np.newaxis, :] # (1, N)
    sigma_col = sigma[:, np.newaxis]
    sigma_row = sigma[np.newaxis, :]
    
    # Difference in means
    diff_mu = mu_row - mu_col # (N, N) -> diff_mu[i, j] = mu[j] - mu[i]
    
    # Difference in variances
    var_sum = sigma_col**2 + sigma_row**2 # (N, N)
    std_diff = np.sqrt(var_sum + 1e-9)
    
    # Z scores
    z_pair = diff_mu / std_diff
    
    # Probabilities P(f_i < f_j)
    p_pair = norm.cdf(z_pair)
    
    # We want the product over j != i
    # Set diagonal to 1 so it doesn't affect the product
    np.fill_diagonal(p_pair, 1.0)
    
    # Log sum to avoid underflow
    log_p_pair = np.log(p_pair + 1e-9)
    
    # Sum over j (axis=1) for each i
    log_p_min = np.sum(log_p_pair, axis=1)
    
    # Convert back to probability (optional, log is also a valid score, higher is better)
    # Using log score is safer and sufficient for ranking
    p_global_min = log_p_min
    
    # Combine with probability of beating f_best
    # Score = P(f_i < f_best) * P(f_i is global min)
    # In log space: log(p_beat_best) + p_global_min
    # But p_beat_best can be 0.
    
    # Let's use the product directly if it's not too small, or just the log sum.
    # Actually, the "Knowledge Gradient" is the expected improvement.
    # This probability product is a proxy for "Likelihood of being the new best".
    
    # Let's just return the log probability of being the global minimum weighted by beating f_best?
    # If we return log_p_min, we are picking the point most likely to be the global min of the candidates.
    # This might ignore f_best if f_best is very good.
    # Example: f_best = -100. All candidates mu=0.
    # p_beat_best is very small.
    # p_global_min is roughly equal for all (symmetric).
    # So we pick a random one. This is fine.
    
    # But if one candidate has mu=-10, sigma=1.
    # p_beat_best is high. p_global_min is high.
    # It will be picked.
    
    # Let's combine them:
    # Score = p_beat_best * exp(p_global_min)
    # To avoid underflow/overflow issues in multiplication, we can work in log space
    # log_score = log(p_beat_best) + p_global_min
    
    log_p_beat_best = np.log(p_beat_best + 1e-9)
    
    scores = log_p_beat_best + p_global_min
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
