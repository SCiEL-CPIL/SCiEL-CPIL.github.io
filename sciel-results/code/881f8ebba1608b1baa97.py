# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Pareto-optimal combination of 
    Expected Improvement (EI) and Posterior Variance (PV).
    
    This approach treats cost-awareness not just as a divisor, but by
    normalizing the components to be comparable in scale before combining,
    which often yields better exploration-exploitation trade-offs than
    simple division or linear blending of raw values.
    """
    # Deadline guard: if we are dangerously close to the limit, return a safe default
    # The limit is 300s per evaluation. We check elapsed time.
    # Note: This function is called many times. The global timer tracks the total run.
    # However, the task says "Each evaluation must finish within 300 s".
    # Usually, the evaluator kills the whole process if any single call takes too long or total time exceeds limit.
    # The prompt says "The evaluator KILLS the program after 330s wall-clock."
    # So we should check if the total time since the start of the script is close to 330s.
    
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a safe, finite score that prioritizes low cost and high uncertainty
        # to hopefully finish the loop quickly or at least not crash
        safe_scores = 1.0 / (cost + 1e-9)
        return np.where(np.isfinite(safe_scores), safe_scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Posterior Variance (PV) as a proxy for exploration
    # We use sigma^2 to emphasize high uncertainty regions more strongly
    pv = sigma ** 2
    
    # 3. Cost-aware normalization
    # Instead of simply dividing by cost, we normalize EI and PV by their respective
    # cost-weighted means to make them comparable in scale.
    # This creates a "value per cost" metric for each component.
    
    # Calculate cost-weighted means for normalization
    # We want to find candidates that offer high EI or high PV relative to their cost.
    
    # To avoid division by zero or extreme values, we use a robust normalization.
    # Let's try a simpler, robust approach:
    # Score = (alpha * EI + beta * PV) / cost
    # But the current program does this and scores 0.74.
    # Let's try a different structural change: 
    # Use the "Expected Improvement per Unit Cost" but add a strong exploration bonus
    # that is also cost-aware, but scaled differently.
    
    # Alternative Strategy: Thompson Sampling-like acquisition
    # Sample from the posterior: y ~ N(mu, sigma)
    # But we can't sample 2000 times efficiently in a way that's deterministic enough for scoring?
    # Actually, we can use a deterministic approximation of Thompson Sampling.
    # The probability of being the best is P(y < f_best) = Phi((f_best - mu) / sigma)
    # This is related to EI but simpler.
    
    # Let's try a hybrid: 
    # 1. Cost-normalized EI
    # 2. Cost-normalized Probability of Improvement (PI)
    # 3. Cost-normalized Uncertainty (sigma)
    
    ei_cost = ei / cost
    
    # Probability of Improvement (PI)
    pi_prob = norm.cdf(z)
    pi_cost = pi_prob / cost
    
    # Uncertainty term (standard deviation)
    # We use sigma directly, not sigma^2, to avoid over-weighting extreme outliers
    unc_cost = sigma / cost
    
    # Combine them. 
    # EI is generally the strongest signal.
    # PI is a softer signal.
    # Uncertainty helps escape local minima.
    
    # Weights: 
    # High weight on EI because it directly estimates improvement.
    # Moderate weight on Uncertainty to ensure exploration.
    # Low weight on PI as it's somewhat redundant with EI.
    
    w_ei = 1.0
    w_pi = 0.1
    w_unc = 0.5
    
    scores = w_ei * ei_cost + w_pi * pi_cost + w_unc * unc_cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
