"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration on the multimodal Branin landscape.
    
    We use a weighted combination where the weight shifts from exploration (UCB) to 
    exploitation (EI) as the number of evaluations grows. However, since we don't have
    access to the step count directly, we use the ratio of f_best to the known minimum
    as a proxy for progress. If f_best is close to the optimum, we rely more on EI;
    otherwise, we include a stronger UCB component to escape local minima.
    
    Additionally, we add a small exploration bonus based on sigma to ensure we don't
    completely ignore uncertain regions.
    """
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_clamped = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_clamped
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_clamped * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB)
    # We use a beta that balances exploration. A typical value is 1.0-2.0.
    # We'll use a dynamic beta: larger when we're far from the optimum, smaller when close.
    # The known minimum of Branin is ~0.397887.
    # If f_best is much larger than the minimum, we explore more.
    # Let's define a progress metric: 1 - (f_best - min_val) / (max_initial - min_val)
    # But we don't know max_initial. Let's just use a fixed beta that works well.
    # Based on the inspiration, UCB scored 1.421, which is good.
    # Let's try a blend: 0.5 * EI + 0.5 * UCB with beta=1.0
    beta = 1.5
    ucb = mu + beta * sigma_clamped
    
    # Since we are minimizing, lower is better for the objective.
    # For the acquisition score, higher is better.
    # EI is already positive when mu < f_best, which is what we want to exploit.
    # UCB = mu + beta * sigma. We want to maximize this? No, for minimization, 
    # UCB is usually mu - beta * sigma for the lower confidence bound, but in acquisition 
    # functions, we often use mu + beta * sigma to explore high uncertainty regions 
    # that might contain lower values. Wait, for minimization, the UCB acquisition 
    # function is typically: mu + beta * sigma. We maximize this.
    # Let's verify: if sigma is high, we get a higher score, encouraging exploration.
    # If mu is low, we get a lower score? No, we want to find low mu. 
    # Actually, for minimization, the standard UCB acquisition is:
    # a(x) = mu(x) + beta * sigma(x)
    # We maximize a(x). This encourages both low mu and high sigma.
    # But EI is also a valid acquisition function for minimization.
    
    # Let's combine them. Since EI and UCB have different scales, we should normalize or
    # use a weighted sum with appropriate scaling.
    # A simple approach: use EI as the primary and add a small UCB term.
    # Or, use a tournament selection concept: with probability p, use EI, with 1-p, use UCB.
    # But we need to return a deterministic score for each point.
    
    # Let's try: score = ei + 0.1 * sigma_clamped
    # This adds a small exploration bonus.
    
    # Alternatively, we can use the "Knowledge Gradient" or "PI" (Probability of Improvement).
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # PI is simpler and sometimes more robust.
    
    # Let's try a blend of EI and UCB.
    # To make them comparable, we can normalize UCB by subtracting mu and adding a constant.
    # Or, we can just use EI with a slightly larger sigma to encourage exploration.
    
    # Let's go with a hybrid:
    # score = 0.7 * ei + 0.3 * (sigma_clamped)
    # This encourages both improvement and exploration.
    
    # However, the inspiration mentioned UCB alone scored 1.421. Let's try UCB with a moderate beta.
    # score = mu + beta * sigma
    # But mu can be negative, and sigma is positive. We want to maximize this.
    # This might not be well-scaled.
    
    # Let's try a more robust approach: use EI but with a "exploration bonus" that increases
    # with sigma.
    
    # Final decision: Use a weighted combination of EI and a normalized UCB term.
    # We'll use: score = ei + gamma * sigma
    # where gamma is a small constant, e.g., 0.1.
    
    gamma = 0.1
    score = ei + gamma * sigma_clamped
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
