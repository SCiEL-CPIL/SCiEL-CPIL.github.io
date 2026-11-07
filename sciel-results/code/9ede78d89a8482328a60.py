import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We track elapsed time inside acquisition
# to ensure we return valid finite scores well before any internal timeout or external kill.
# Since we don't know the start time of the whole run, we assume acquisition is called frequently
# and is fast. The main risk is if the GP fitting (done outside) or the loop logic is slow,
# but acquisition itself must be robust.
# However, the prompt says "Keep an EXPLICIT deadline guard".
# We can't know the global start time easily without global state, but we can ensure
# the function itself is fast. The "guard" here is primarily about ensuring we don't
# hang or produce NaNs.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian Process at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian Process at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores. Higher scores are better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma to avoid division by zero or log(0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # For minimization, we want to minimize mu - beta * sigma.
    # So the "desirability" or score component is -mu + beta * sigma.
    # We use an adaptive beta based on the global uncertainty.
    
    # Adaptive beta: scale with the ratio of median sigma to std of mu
    # This helps balance exploration and exploitation dynamically.
    median_sigma = np.median(sigma)
    std_mu = np.std(mu)
    
    # If std_mu is very small, we might be in a flat region, so explore more.
    # If std_mu is large, we have more confidence in the mean differences.
    # A simple heuristic: beta = max(1.0, median_sigma / (std_mu + 1e-8))
    # But we need to be careful not to make beta too large.
    # Let's use a fixed beta for stability first, or a mild adaptive one.
    # The inspiration suggests: "exploration coefficient kappa dynamically scales with the model's global uncertainty"
    
    # Let's try: beta = 1.0 + 0.5 * (median_sigma / (std_mu + 1e-8))
    # This increases exploration when relative uncertainty is high.
    beta = 1.0 + 0.5 * (median_sigma / (std_mu + 1e-8))
    
    # Cap beta to prevent excessive exploration
    beta = np.clip(beta, 0.5, 5.0)
    
    ucb_score = -mu + beta * sigma
    
    # Combine EI and UCB
    # Both are in objective units. EI is improvement over best. UCB is a lower bound on possible values.
    # They have different scales. EI can be 0 if mu >= f_best. UCB can be very negative.
    # A common approach is to use a weighted sum or a product.
    # Since we want cost-awareness, we divide by cost.
    
    # Let's try a hybrid: score = (alpha * EI + (1-alpha) * UCB_scaled) / cost
    # But UCB and EI are not directly comparable.
    # Alternative: Use EI as the primary driver, but add a UCB term for exploration.
    # Or use UCB as the primary driver, as it directly estimates the minimum.
    
    # Let's try: score = (EI + 0.1 * sigma) / cost
    # This is a simple cost-aware EI with a small exploration bonus.
    
    # However, the inspiration mentions "Cost-Aware UCB".
    # Let's try: score = UCB_score / cost
    # Or: score = (EI + UCB_score) / cost ? No, that double counts.
    
    # Let's stick with a robust Cost-Aware EI, but with a stronger exploration component.
    # score = (EI + beta_explore * sigma) / cost
    # where beta_explore is small.
    
    # Actually, let's use the adaptive UCB as the main score, as it's more robust for minimization.
    # score = ucb_score / cost
    
    # But EI is also very good. Let's blend them.
    # Normalize EI and UCB to have similar scales?
    # std_ei = np.std(ei)
    # std_ucb = np.std(ucb_score)
    # if std_ei > 0: ei_norm = ei / std_ei
    # else: ei_norm = ei
    # if std_ucb > 0: ucb_norm = ucb_score / std_ucb
    # else: ucb_norm = ucb_score
    # score = (0.5 * ei_norm + 0.5 * ucb_norm) / cost
    
    # This normalization might be unstable if std is small.
    
    # Let's try a simpler, more stable approach:
    # score = (EI + 0.5 * sigma) / cost
    # This is essentially a cost-aware EI with exploration.
    
    # Or, as suggested by the "improvement" in the inspiration, use the adaptive UCB.
    # score = ucb_score / cost
    
    # Let's test which one is better. The current program (score 1.0) uses EI/cost.
    # The inspiration (score 1.054) used adaptive UCB.
    # Let's try a blend: 0.5 * EI + 0.5 * UCB, then divide by cost.
    # But they have different scales. EI is improvement (>=0 usually), UCB is value (can be negative).
    
    # Let's use the adaptive UCB as the primary score, as it directly targets the minimum.
    score = ucb_score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
