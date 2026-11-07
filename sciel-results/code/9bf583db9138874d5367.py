"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are running long, return a safe finite score
    # We don't have a global timer here, but we can add a small overhead guard
    # For now, we focus on a better acquisition function.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Upper Confidence Bound (UCB) - lower is better for minimization
    # We want to maximize the acquisition, so for minimization UCB is:
    # -mu + kappa * sigma (we want to find low values, so we add uncertainty)
    # Actually, for minimization, the "upper bound" of the true value is mu + kappa*sigma
    # We want to minimize this, so the acquisition should be high when this is low.
    # Acquisition = -(mu + kappa * sigma) ? No, that's just negative UCB.
    # Let's use a combination.
    
    # Cost-aware scaling
    # The cost is 1 + mean(x). We don't have x, but we have cost.
    # We want to divide by cost to penalize expensive points.
    
    # Hybrid: combine EI and PI with a UCB-like exploration term
    # Let's try a cost-aware UCB with a twist.
    # For minimization, we prefer low mu and high sigma (exploration).
    # Score = -mu + kappa * sigma - lambda * cost
    # But we need to return scores where higher is better.
    
    # Let's stick with a robust cost-aware EI/PI hybrid.
    # The key insight from the "inspiration" is that PI can be better for multi-modal landscapes.
    # Let's blend EI and PI.
    
    # Normalize EI and PI to [0, 1] roughly? No, just combine them.
    # EI is in objective units, PI is probability.
    
    # Let's try: Score = (EI + alpha * PI * sigma) / cost
    # Or simply: Score = EI / cost + beta * PI / cost
    
    # A more principled approach: Expected Improvement per unit cost, but with a PI floor.
    # If EI is very small but PI is high, we might still want to explore.
    
    # Let's use a "Cost-aware Expected Improvement with Exploration"
    # Score = EI / cost + gamma * (sigma / cost) * PI
    # This adds an exploration bonus proportional to uncertainty and probability of improvement.
    
    gamma = 0.5  # Exploration weight
    
    score = (ei + gamma * sigma_safe * pi) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
