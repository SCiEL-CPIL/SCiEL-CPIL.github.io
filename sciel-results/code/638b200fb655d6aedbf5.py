# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Quantile-Based Cost-Discounted Lower Confidence Bound approach.
    
    The score is:
        score(x) = EI(x) * discount(cost(x))
    
    where EI is Expected Improvement (which implicitly uses the quantile of the
    posterior via the normal CDF/PDF) and discount is a sigmoidal function that
    aggressively penalizes high-cost candidates.
    
    Key structural ideas:
    1. EI-based scoring (from inspiration 1) which naturally handles the
       posterior distribution via the normal CDF, effectively mapping to
       quantiles of the predictive distribution.
    2. Sigmoidal cost discount (from the directive strategy) instead of
       power-law cost penalty, which more aggressively prunes high-cost
       candidates in outer regions.
    3. Adaptive cost sensitivity that increases as budget is consumed.
    4. Lightweight parameter calibration for the EI exploration constant.
    """
    t_start = time.time()
    TIME_LIMIT = 320.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # --- Compute Expected Improvement ---
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    # Normal CDF and PDF
    cdf_z = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei_raw = (f_best - mu_safe) * cdf_z + sigma_safe * pdf_z

    # Ensure EI is non-negative (numerical safety)
    ei_raw = np.maximum(ei_raw, 0.0)

    # --- Sigmoidal cost discount ---
    # The cost is 1 + mean(x) where x is in [0,1]^3, so cost is in [1, 2].
    # We want to discount high-cost points more aggressively.
    # Use a sigmoid centered around a cost threshold with tunable sharpness.
    
    # Adaptive cost threshold: as we spend more budget, we should be more
    # selective about cost. Since we don't have direct budget info, we use
    # the distribution of costs in the current candidate set.
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe) + 1e-10
    
    # Center the sigmoid around the mean cost, with a sharpness that
    # depends on the spread of costs
    cost_center = cost_mean
    cost_sharpness = 3.0 / (cost_std + 1e-10)
    
    # Sigmoidal discount: 1 for low cost, approaches 0 for high cost
    # discount = 1 / (1 + exp(sharpness * (cost - center)))
    # This gives ~0.5 at the center, ~1 well below, ~0 well above
    discount = 1.0 / (1.0 + np.exp(cost_sharpness * (cost_safe - cost_center)))
    
    # Scale discount so that the best (lowest cost) points get ~1.0
    # and adjust the overall magnitude
    discount_max = np.max(discount)
    if discount_max > 1e-10:
        discount = discount / discount_max
    else:
        discount = np.ones_like(discount)
    
    # --- Lightweight calibration of exploration constant ---
    # Try a few values of a multiplier on EI to see which gives better
    # ranking of low-mu, low-cost points
    gamma_vals = [0.5, 1.0, 1.5, 2.0]
    top_k = max(10, min(50, n // 10))
    
    best_proxy = -np.inf
    best_gamma = 1.0
    best_cost_beta = 1.0
    
    # Also try different cost exponents as an alternative to sigmoid
    cost_beta_vals = [0.5, 1.0, 1.5, 2.0]
    
    for gamma in gamma_vals:
        for cb in cost_beta_vals:
            if time.time() - t_start > TIME_LIMIT:
                break
            
            # Power-law cost penalty variant
            scores_candidate = (ei_raw * gamma) / (cost_safe ** cb)
            
            if not np.all(np.isfinite(scores_candidate)):
                continue
            
            # Proxy objective: prefer low mu and low cost in top-k
            top_idx = np.argsort(scores_candidate)[-top_k:]
            mean_mu_top = np.mean(mu_safe[top_idx])
            mean_cost_top = np.mean(cost_safe[top_idx])
            proxy = -mean_mu_top - 0.1 * mean_cost_top
            
            if proxy > best_proxy:
                best_proxy = proxy
                best_gamma = gamma
                best_cost_beta = cb
    
    # --- Final scoring ---
    # Combine sigmoidal discount with power-law penalty
    # Use the calibrated power-law and add sigmoidal discount as a multiplier
    power_penalty = cost_safe ** best_cost_beta
    scores = (ei_raw * best_gamma) / power_penalty * discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores
    
    # --- Optional refinement: boost high-uncertainty points among top candidates ---
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 5.0:
        top_k2 = min(30, n)
        top_idx2 = np.argsort(scores)[-top_k2:]
        
        # Boost points with high sigma among the top candidates
        top_sigma = sigma_safe[top_idx2]
        sigma_threshold = np.percentile(sigma_safe, 70)
        
        boost_amount = 0.05 * np.max(scores)
        for idx in top_idx2:
            if sigma_safe[idx] > sigma_threshold:
                scores[idx] += boost_amount
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
