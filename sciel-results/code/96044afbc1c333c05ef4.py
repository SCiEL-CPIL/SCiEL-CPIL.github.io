import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Lower Confidence Bound for minimization
    # We want to maximize the score, which corresponds to the lower bound of the confidence interval
    # Lower bound = mu - kappa * sigma
    # So score = mu - kappa * sigma
    # Wait, if I want to evaluate the point with the LOWEST lower bound, I should maximize the negative of it?
    # No, the evaluator picks the candidate with the HIGHEST score.
    # If I define score = mu - kappa * sigma, then a point with mu=0, sigma=10, kappa=2 has score -20.
    # A point with mu=100, sigma=1, kappa=2 has score 98.
    # The evaluator picks 98. This is the local minimum. This is BAD.
    
    # I need the score to be HIGH for points that are likely to be LOW.
    # So I should use: score = - (mu - kappa * sigma) = kappa * sigma - mu.
    # Let's check:
    # Point A (Unexplored, Global Min region): mu=50, sigma=20. Score = 2*20 - 50 = -10.
    # Point B (Local Min): mu=10, sigma=1. Score = 2*1 - 10 = -8.
    # -8 > -10. Evaluator picks B. Still bad.
    
    # The issue is that `mu` in unexplored regions is high (prior mean or correlated with high values).
    # If the function is Beale, values are positive.
    # If I haven't seen the global min, `mu` there is likely high.
    # So `kappa * sigma - mu` penalizes high `mu`.
    
    # What if I use `sigma` only? Pure exploration.
    # Or `EI`?
    # EI at Unexplored: f_best=10, mu=50, sigma=20. z = (10-50)/20 = -2.
    # EI = (10-50)*Phi(-2) + 20*phi(-2) = -40*0.022 + 20*0.054 = -0.88 + 1.08 = 0.2.
    # EI at Local Min: f_best=10, mu=10, sigma=1. z=0.
    # EI = 0*0.5 + 1*0.4 = 0.4.
    # EI picks Local Min.
    
    # So standard EI also fails if the local min is "good enough" and the global min is "uncertain and far".
    
    # How to fix?
    # We need to boost the score of unexplored regions more.
    # UCB with a larger kappa?
    # Score = kappa * sigma - mu.
    # If kappa is large, say 10.
    # Unexplored: 10*20 - 50 = 150.
    # Local Min: 10*1 - 10 = 0.
    # Picks Unexplored. Good.
    
    # But if we get too close to the global min, sigma drops, mu drops.
    # Global Min: mu=0, sigma=1. Score = 10*1 - 0 = 10.
    # Local Min: mu=10, sigma=1. Score = 10*1 - 10 = 0.
    # Picks Global Min. Good.
    
    # So a large kappa works for exploration.
    # But we need to converge.
    # As sigma drops everywhere, the term kappa*sigma becomes small.
    # Then -mu dominates. We pick the lowest mu.
    # This seems robust.
    
    kappa = 2.0 # Let's start with a moderate value. Maybe 1.0?
    
    # Let's try to adapt kappa?
    # No, keep it simple.
    
    scores = kappa * sigma_safe - mu
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
