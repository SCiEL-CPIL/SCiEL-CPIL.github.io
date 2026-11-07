import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a modified Expected Improvement (EI) with a diversity penalty.
    """
    # Ensure numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Since we want to maximize the score, higher EI is better.
    # Note: Standard EI is 0 when mu > f_best. We want to encourage exploration 
    # even when mu > f_best if sigma is high, but standard EI handles this via 
    # the sigma*phi(z) term which is small for large negative z (mu >> f_best).
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Diversity/Repulsion Term:
    # We want to penalize points that are "close" to the current best in the 
    # predicted space or similar in structure. 
    # Since we don't have x coordinates, we use a heuristic based on the 
    # relationship between mu and sigma.
    # A common proxy for "explored" or "similar" in absence of x is 
    # high correlation with the current best state, but we don't have that.
    
    # Alternative Strategy: 
    # The Weierstrass function is multimodal. With only 30 evaluations, 
    # greedy EI often gets stuck in the first local minimum found.
    # We can introduce a "novelty" score based on the uncertainty profile.
    # However, without x, we cannot compute distance to previous points.
    
    # Let's look at the "inspiration" data.
    # p12 (EI) scored 1.0 (worse than UCB 2.15).
    # p9 (UCB beta=2.5) scored 2.15.
    # The UCB with high beta (2.5) is very exploratory.
    # Maybe the issue is that UCB explores too much? Or not enough?
    # EI is generally more exploitation-focused.
    
    # A different family: **Knowledge Gradient (KG)** approximation?
    # Or **Lower Confidence Bound (LCB)**? No, that's just UCB with sign flip.
    
    # Let's try **Pareto-EI** or a **Hybrid** approach.
    # Actually, a very different strategy is **Thompson Sampling**.
    # But we only get mu and sigma, not the full GP posterior for sampling.
    # We can approximate Thompson Sampling by sampling from N(mu, sigma^2)
    # and picking the min. But we must return scores for all 2000 points.
    # We can compute the probability of being the best.
    
    # Let's try **Probability of Improvement (PI)** with a heavy exploration boost?
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Let's try a **Weighted Sum of EI and UCB** but with dynamic beta.
    # Or better: **Look-ahead** is not possible.
    
    # Let's reconsider the failure of EI (score 1.0).
    # EI is 0 when mu is much larger than f_best.
    # UCB = -mu + beta*sigma. If mu is large (bad), -mu is very negative (bad score).
    # So both punish high mu.
    # Why did EI fail? Maybe it was too conservative.
    # UCB with beta=2.5 was good.
    
    # What if we use **UCB with a non-linear sigma boost**?
    # Or **UCB with a floor**?
    
    # Let's try a **Different Design Family**: **Bayesian Optimization with a Repulsive Term in the Score**.
    # Since we don't have x, we can't do spatial repulsion.
    
    # How about **Max-Value Entropy Search (MES)** approximation?
    # Not feasible with just mu/sigma.
    
    # Let's look at the structure of Weierstrass.
    # It has many local minima.
    # The key is to jump between basins.
    # UCB with high beta does this by picking high sigma regions.
    # As the GP fits the data, sigma drops in explored regions.
    # So UCB naturally moves to unexplored regions.
    
    # Perhaps the current UCB is *too* exploratory and misses the global minimum?
    # Or is it *not* exploratory enough?
    # Score 2.15 is better than baseline 1.0.
    
    # Let's try **Expected Hypervolume Improvement (EHVI)**? No, single objective.
    
    # Let's try **UCB with an adaptive beta**.
    # Start with high beta (exploration) and decrease it (exploitation) as we get more evaluations?
    # But we don't know the step number. We only have mu, sigma, f_best.
    # We can infer progress from f_best? If f_best is close to 0, we are doing well.
    # Weierstrass min is 0.
    # If f_best is small, we should exploit. If f_best is large, we should explore.
    
    # Strategy: **Adaptive UCB based on f_best**.
    # beta = base_beta * (f_best / scale)
    # If f_best is high (early), beta is high.
    # If f_best is low (late), beta is low.
    
    # This is a materially different strategy: **State-Dependent Acquisition**.
    
    # Let's calibrate.
    # Weierstrass 1D range is roughly [0, 10] or so? 
    # The minimum is 0.
    # Typical random start might be ~1-5.
    
    # Let's try:
    # beta = 1.0 + 5.0 * np.exp(-f_best)
    # If f_best = 0, beta = 6.0 (high exploration? Wait, if f_best is good, we want exploitation).
    # If f_best = 0, we are near the optimum. We should exploit.
    # So beta should be LOW when f_best is LOW.
    # beta = 0.5 + 2.0 * np.exp(-f_best * 5.0) ?
    
    # Let's think about the trade-off.
    # Early on: f_best is high (e.g., 5.0). We want to explore. Beta should be high.
    # Late on: f_best is low (e.g., 0.1). We want to exploit. Beta should be low.
    
    # So beta should decrease as f_best decreases.
    # beta = max_beta * (f_best / max_expected_f) ?
    # Or beta = max_beta * np.exp(-f_best) ?
    # If f_best = 5, exp(-5) is small. Beta small. This is wrong. We want high beta early.
    
    # Correct logic:
    # High f_best (bad) -> High Beta (Explore)
    # Low f_best (good) -> Low Beta (Exploit)
    
    # So beta should be proportional to f_best.
    # beta = k * f_best
    # But if f_best is very small, beta becomes 0. Pure exploitation.
    # This might get stuck.
    # Let's add a floor.
    # beta = 0.5 + 2.0 * f_best
    
    # Let's test this logic.
    # Initial f_best ~ 5. beta = 0.5 + 10 = 10.5. Very exploratory.
    # Later f_best ~ 0.5. beta = 0.5 + 1 = 1.5. Moderate.
    # Final f_best ~ 0.01. beta = 0.5 + 0.02 = 0.52. Exploitative.
    
    # This seems like a sound, materially different strategy (Adaptive UCB).
    
    # Implementation:
    # Use a soft cap to prevent beta from exploding if f_best is weird.
    # f_best is the minimum value. It should be positive.
    
    beta = 0.5 + 2.0 * np.clip(f_best, 0.0, 10.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
