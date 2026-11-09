"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 310s safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition for Weierstrass 1D minimization.
    
    Combines:
    1. Expected Improvement (EI) for exploitation
    2. UCB-like uncertainty bonus for exploration
    3. Diversity penalty based on pairwise kernel correlation among top candidates
    """
    if _check_deadline():
        # Fallback: simple EI to avoid timeout
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    n = len(mu)
    eps = 1e-9
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # --- Component 1: Expected Improvement (exploitation) ---
    z = (f_best - mu) / sigma_safe
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- Component 2: Uncertainty bonus (exploration) ---
    # Scale uncertainty relative to the range of observed values
    mu_range = np.ptp(mu) + eps
    ucb_bonus = 0.5 * sigma_safe / (mu_range + eps)
    
    # --- Component 3: Diversity penalty ---
    # We need to penalize candidates that are too close to each other in input space.
    # Since we don't have coordinates, we approximate diversity using the kernel structure.
    # The Matern-5/2 kernel with length scale 0.2 on [0,1] implies that points within
    # ~0.2 are highly correlated. We can estimate pairwise correlation from the GP posterior.
    # 
    # However, without coordinates, we cannot directly compute the kernel. 
    # Instead, we use a proxy: candidates with very low sigma are likely in well-explored
    # regions, while those with high sigma are in unexplored regions. We penalize clustering
    # of high-EI candidates by looking at the distribution of mu values.
    #
    # A practical approach: rank candidates by (ei + ucb_bonus) and penalize the top-K
    # based on their mutual similarity. Since we lack coordinates, we use the fact that
    # in 1D, if two points have very similar mu and sigma, they are likely close.
    #
    # Simpler and more robust: use a penalty based on the density of candidates in the
    # (mu, sigma) space. Candidates in dense regions get penalized.
    
    # Compute base score
    base_score = ei + ucb_bonus
    
    # Diversity penalty: penalize candidates that are in the same "region" as other high-scoring candidates
    # We use a simple density-based penalty: for each candidate, count how many other candidates
    # have similar mu and sigma values (within a tolerance).
    
    # To keep it efficient, we subsample for the diversity calculation
    n_sample = min(n, 500)
    if n_sample < n:
        idx = np.random.choice(n, n_sample, replace=False)
        mu_s = mu[idx]
        sigma_s = sigma_safe[idx]
        base_s = base_score[idx]
    else:
        mu_s = mu
        sigma_s = sigma_safe
        base_s = base_score
    
    # Compute pairwise similarity in (mu, sigma) space
    # Similarity = exp(-((mu_i - mu_j)^2 + (sigma_i - sigma_j)^2) / (2 * scale^2))
    mu_scale = np.std(mu_s) + eps
    sigma_scale = np.std(sigma_s) + eps
    scale = max(mu_scale, sigma_scale)
    
    # For efficiency, compute penalty only for top candidates
    top_k = min(100, n_sample)
    top_idx_local = np.argsort(base_s)[-top_k:]
    
    penalty = np.zeros(n_sample)
    for i in range(top_k):
        ii = top_idx_local[i]
        # Compute similarity to other top candidates
        dm = (mu_s[ii] - mu_s) / (mu_scale + eps)
        ds = (sigma_s[ii] - sigma_s) / (sigma_scale + eps)
        dist2 = dm**2 + ds**2
        # Exclude self
        dist2[ii] = np.inf
        # Sum of similarities to other top candidates
        sim = np.exp(-dist2 / 2.0)
        penalty[ii] = np.sum(sim)
    
    # Apply penalty: subtract a fraction of the penalty from the score
    penalty_weight = 0.3
    base_score_adj = base_score - penalty_weight * penalty
    
    # For candidates not in the top-k, penalty is 0
    if n_sample < n:
        # Map back to full array
        full_adj = base_score.copy()
        full_adj[idx[top_idx_local]] = base_score_adj[top_idx_local]
        return full_adj
    else:
        return base_score_adj
# EVOLVE-BLOCK-END
