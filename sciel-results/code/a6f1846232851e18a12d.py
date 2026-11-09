# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Lower Confidence Bound (LCB).
    
    Instead of EI (which can converge prematurely), we use an LCB-style score
    that dynamically balances exploitation and exploration:
        score(x) = -mu(x) - beta(t) * sigma(x) - lambda * cost(x)
    
    where beta(t) adapts based on the relative uncertainty in the current
    posterior, encouraging exploration when the GP is uncertain and exploitation
    when it is confident. The cost term linearly penalizes expensive evaluations.
    
    Key structural differences from EI-based approaches:
    1. LCB directly targets the minimum, not improvement over f_best
    2. Adaptive beta scales with the ratio of max sigma to current sigma
    3. Linear cost penalty (not power law) for simpler, more stable behavior
    4. No internal calibration loop - parameters are set analytically
    """
    t_start = time.time()
    TIME_LIMIT = 325.0  # Safety margin before 330s kill
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # --- Adaptive LCB computation ---
    
    # Compute statistics of the current posterior
    mu_min = np.min(mu_safe)
    mu_max = np.max(mu_safe)
    mu_range = mu_max - mu_min + 1e-10
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    
    # Adaptive exploration parameter:
    # When uncertainty is high relative to the function range, explore more.
    # beta scales with the ratio of mean sigma to the function range.
    # We also add a base exploration to avoid pure exploitation.
    uncertainty_ratio = sigma_mean / (mu_range + 1e-10)
    
    # Base exploration constant, scaled by uncertainty
    # Higher uncertainty -> more exploration
    base_beta = 1.0
    adaptive_factor = 1.0 + 2.0 * uncertainty_ratio
    
    # The effective beta for each point: points with higher sigma get more
    # exploration credit. This creates a dynamic balance.
    # beta_i = base_beta * adaptive_factor * (sigma_i / sigma_mean)
    # But cap it to avoid extreme values
    sigma_ratio = sigma_safe / (sigma_mean + 1e-10)
    sigma_ratio = np.clip(sigma_ratio, 0.1, 5.0)
    
    beta_i = base_beta * adaptive_factor * sigma_ratio
    
    # Cost penalty: linear, normalized by mean cost
    cost_mean = np.mean(cost_safe)
    cost_normalized = cost_safe / (cost_mean + 1e-10)
    
    # Weight for cost penalty: small enough to not dominate, but significant
    # enough to prefer cheaper points when mu/sigma are similar
    lambda_cost = 0.3
    
    # LCB score: we want to MINIMIZE mu + beta*sigma + lambda*cost
    # But acquisition should be MAXIMIZED, so we negate
    lcb_value = mu_safe + beta_i * sigma_safe + lambda_cost * cost_normalized
    scores = -lcb_value
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        # If any -inf remain, replace with a very negative finite value
        scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Time check: if we're running out of time, return immediately
    if time.time() - t_start > TIME_LIMIT:
        return scores
    
    # --- Optional refinement: local search around top candidates ---
    # We can afford a lightweight refinement to break ties or improve ranking
    # among the top candidates. This is a structural addition: we use the
    # GP posterior to identify regions of high uncertainty and boost exploration
    # there.
    
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 5.0:
        # Identify top 100 candidates
        top_k = min(100, n)
        top_idx = np.argsort(scores)[-top_k:]
        
        # For these top candidates, check if they are in high-uncertainty regions
        # and give them a small boost if so
        top_sigma = sigma_safe[top_idx]
        top_mu = mu_safe[top_idx]
        
        # Points that are both low-mu and high-sigma are ideal: they might
        # be near the optimum but we're not sure yet. Boost them slightly.
        sigma_percentile = np.percentile(sigma_safe, 75)
        high_uncertainty = top_sigma > sigma_percentile
        
        # Small boost for high-uncertainty points among the top candidates
        boost = 0.05 * sigma_mean * high_uncertainty
        scores[top_idx] += boost
    
    # Final safety: ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
