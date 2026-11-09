import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function: EI + Relative Uncertainty Bonus - Mean Penalty.
    
    Designed for minimization of the Beale function.
    - EI handles exploitation.
    - Relative uncertainty (sigma / max(sigma)) encourages exploration in 
      high-uncertainty regions, scaled to be significant.
    - Mean penalty discourages exploring regions where the GP predicts 
      high function values, even if uncertainty is high.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Relative Uncertainty Bonus
    # We want to explore where sigma is high relative to the scale of the problem.
    # The Beale function has a global minimum of 0 and can reach values > 100.
    # f_best starts around 100+ and decreases.
    # Using a relative measure helps avoid being dominated by early high sigma values
    # or becoming too greedy later.
    
    # Scale factor: We want the exploration term to be comparable to EI initially.
    # A common heuristic is to use the standard deviation of the GP predictions 
    # or a fraction of the current best value.
    # Let's use a robust scale based on the data provided in this call.
    # max_sigma represents the highest uncertainty in the current candidate set.
    max_sigma = np.max(sigma)
    
    # If max_sigma is 0, exploration is 0.
    # We use a power law to make the bonus non-linear.
    # (sigma / max_sigma) ** alpha
    # alpha < 1 makes the bonus less concentrated on the single max point,
    # spreading exploration more. alpha > 1 makes it more concentrated.
    # Let's try alpha = 0.5 for a softer spread.
    rel_unc = (sigma / max_sigma) ** 0.5
    
    # Weight for exploration. 
    # In early stages, we want more exploration. In later stages, less.
    # However, we don't have step count directly. 
    # We can use f_best as a proxy. As f_best gets closer to 0 (the min), 
    # we should rely more on EI.
    # But f_best is the value, not the regret.
    # Let's use a fixed weight that balances well for Beale.
    # Based on inspiration: "EI plus relative-uncertainty exploration bonus"
    # Let's set the weight such that at the start, exploration is significant.
    # If f_best is large (e.g., 100), EI values can be large.
    # Let's try a weight of 0.5 * f_best? No, that scales too much.
    # Let's try a fixed weight relative to the scale of the function.
    # Beale function values are roughly in [0, 100+].
    # Let's use a weight that makes the exploration term comparable to EI.
    
    # Alternative: Use UCB-style but with relative sigma.
    # Score = EI + kappa * sigma
    # The inspiration mentions "relative-uncertainty exploration bonus".
    # Let's use: bonus = w * (sigma / max_sigma) * scale
    # What scale? Maybe sqrt(f_best)? Or just a constant?
    # Let's look at the "measured" improvement: 1.797 -> 6.334.
    # This suggests a significant structural change helped.
    
    # Let's try: Score = EI + 0.5 * sigma + penalty
    # But the inspiration says "relative".
    
    # Let's try: Score = EI + 1.0 * (sigma / max_sigma) * (f_best - min_possible)
    # min_possible is 0.
    # So: Score = EI + 1.0 * (sigma / max_sigma) * f_best
    
    # This means if f_best is 100, the max exploration bonus is 100 * 1.0 = 100.
    # EI max is roughly f_best (if mu is very low and sigma is 0, EI is f_best - mu).
    # So this balances exploration and exploitation nicely in early stages.
    # As f_best drops to 1, the max bonus drops to 1, making the function more greedy.
    
    exploration_weight = 1.0
    exploration_bonus = exploration_weight * rel_unc * f_best
    
    # 3. Mean Penalty
    # "mild penalty for high predicted values"
    # If mu is very high, we shouldn't pick it even if sigma is high.
    # Penalty = lambda * max(0, mu - f_best)
    # This ensures we don't explore regions predicted to be worse than current best.
    # However, EI is already 0 when mu > f_best (approximately, actually EI becomes negative if we don't clip, but standard EI is max(0, ...)).
    # Wait, standard EI for minimization is (f_best - mu)Phi(z) + sigma phi(z).
    # If mu > f_best, z is negative. Phi(z) is small. 
    # Actually, if mu >> f_best, EI becomes negative?
    # Let's check: f_best=10, mu=20, sigma=1. z=(10-20)/1 = -10. Phi(-10)~0. phi(-10)~0. EI ~ 0.
    # So EI naturally penalizes high mu.
    # But adding an explicit penalty might help "avoid wasting evaluations on obviously poor candidates" more aggressively.
    
    # Let's add a soft penalty: -0.1 * (mu - f_best) for mu > f_best
    # Or just rely on EI?
    # The inspiration says "mild penalty".
    # Let's use: penalty = 0.1 * np.maximum(0, mu - f_best)
    
    penalty_weight = 0.1
    penalty = penalty_weight * np.maximum(0, mu - f_best)
    
    scores = ei + exploration_bonus - penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
