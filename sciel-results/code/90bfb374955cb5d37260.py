import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI) 
    and Upper Confidence Bound (UCB) for minimization.
    
    Args:
    mu: (N,) array of GP mean predictions in objective units.
    sigma: (N,) array of GP standard deviations in objective units.
    f_best: float, the lowest observed value so far.
    cost: (N,) array of costs for each candidate.
    
    Returns:
    scores: (N,) array of acquisition scores.
    """
    epsilon = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (for minimization)
    # We want to explore where the function could be lower.
    # LCB = mu - beta * sigma. 
    # To maximize score, we want to minimize LCB, so we maximize -LCB = -mu + beta * sigma.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # The inspiration suggests 70/30 blend.
    # We need to be careful with scaling. EI and UCB are in objective units.
    # If f_best is around -3 and mu is around -3, EI is small.
    # UCB term -mu is around 3.
    # This might make UCB dominate.
    # Let's normalize or adjust weights.
    
    # Alternative: Use Probability of Improvement (PI) which is bounded [0,1]?
    # PI = Phi(z).
    # But EI is generally better.
    
    # Let's try to scale UCB to be comparable to EI.
    # Or just use a smaller weight for UCB if it's too large.
    # However, the inspiration worked with a blend.
    
    # Let's use a weight of 0.7 for EI and 0.3 for UCB.
    # If UCB is much larger, it might dominate.
    # Let's check magnitudes.
    # If f_best = -3, mu = -3, sigma = 0.1.
    # z = 0. EI = 0.1 * 0.4 = 0.04.
    # UCB = 3 + 0.1 = 3.1.
    # 0.7 * 0.04 + 0.3 * 3.1 = 0.028 + 0.93 = 0.958.
    # If mu = -3.1, sigma = 0.1.
    # z = 1. EI = 0.1 * 0.84 + 0.1 * 0.24 = 0.108.
    # UCB = 3.1 + 0.1 = 3.2.
    # Score = 0.7 * 0.108 + 0.3 * 3.2 = 0.0756 + 0.96 = 1.0356.
    # The UCB term is driving the score. This is essentially a UCB search with a small EI correction.
    # This is good for exploration.
    
    blended_score = 0.7 * ei + 0.3 * ucb
    
    # Cost-aware
    scores = blended_score / (cost + epsilon)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
