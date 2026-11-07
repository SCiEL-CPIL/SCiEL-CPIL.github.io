import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Structural change: Cost-aware UCB with asymmetric cost discounts
    # UCB: -mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Cost-aware: discount exploitation more than exploration for expensive points
    
    # Cost is in [1, 2]. Let c = cost - 1, so c in [0, 1]
    c = cost - 1.0
    
    # Asymmetric cost discounts:
    # Exploitation (mu) gets a stronger discount: factor = 1 / (1 + alpha_exp * c)
    # Exploration (sigma) gets a weaker discount: factor = 1 / (1 + alpha_explore * c)
    # This allows expensive regions to still be explored but penalizes exploiting them
    
    alpha_exp = 0.15   # stronger discount on exploitation
    alpha_explore = 0.05  # weaker discount on exploration
    
    cost_factor_exp = 1.0 / (1.0 + alpha_exp * c)
    cost_factor_explore = 1.0 / (1.0 + alpha_explore * c)
    
    # UCB term for minimization: -mu + beta * sigma
    # But we need to scale appropriately. Let's use a beta that adapts.
    # Use beta = 0.5 as a reasonable exploration-exploitation balance
    beta = 0.5
    
    ucb = -mu * cost_factor_exp + beta * sigma_safe * cost_factor_explore
    
    # Combine EI and UCB with a weighted sum
    # EI captures expected improvement, UCB captures uncertainty
    # Weight them: 0.7 * EI + 0.3 * UCB_normalized
    # Normalize UCB to be comparable to EI
    # EI is in [0, ~1], UCB can be negative or positive
    # Let's just use EI as the primary and add a small UCB bonus
    
    # Actually, let's use a different structural approach:
    # Cost-aware EI where the improvement threshold is adjusted by cost
    # For expensive points, require more expected improvement to justify the cost
    
    # Alternative: use the EI but scale it by a cost-dependent factor
    # factor = 1 / (1 + lambda * c) where lambda controls cost sensitivity
    
    # Let me try: score = EI * cost_discount + exploration_bonus * cost_discount_light
    # where cost_discount = 1 / (1 + lambda * c)
    
    lambda_main = 0.12
    lambda_expl = 0.04
    
    cost_discount = 1.0 / (1.0 + lambda_main * c)
    cost_discount_expl = 1.0 / (1.0 + lambda_expl * c)
    
    # Exploration bonus: use sigma directly as a proxy for uncertainty
    # Scale it to be comparable to EI
    expl_bonus = 0.3 * sigma_safe
    
    scores = ei * cost_discount + expl_bonus * cost_discount_expl
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
