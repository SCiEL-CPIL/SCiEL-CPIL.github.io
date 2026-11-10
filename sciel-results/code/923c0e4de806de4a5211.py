import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with a relaxed target.
    
    Parameters:
    mu: (2000,) GP mean
    sigma: (2000,) GP standard deviation
    f_best: float, best observed value
    cost: (2000,) cost of evaluating each candidate
    
    Returns:
    scores: (2000,) acquisition scores, higher is better
    """
    # Relax the target to maintain exploration pressure
    # If f_best is negative, 0.9 * f_best is closer to 0 (worse), making it easier to "beat"
    # If f_best is positive, 0.9 * f_best is closer to 0 (better), but wait...
    # If f_best > 0, minimizing means we want lower values.
    # If f_best = 10, target = 9. We want f(x) < 9.
    # Original EI: f(x) < 10.
    # Target 9 is stricter? No, 9 < 10.
    # If we require f(x) < 9, it is HARDER to satisfy than f(x) < 10.
    # So z = (9 - mu) / sigma is SMALLER than (10 - mu) / sigma.
    # This DECREASES EI.
    
    # Let's re-read the hint: "prevents the acquisition function from collapsing to zero... maintaining exploration pressure"
    # If f_best is close to min, EI collapses to 0.
    # To prevent collapse, we need to INCREASE EI scores.
    # To increase EI, we need to make the target EASIER to beat.
    # If f_best = 10 (positive), easier target is 11 (worse).
    # If f_best = -10 (negative), easier target is -9 (worse, i.e., larger).
    
    # So, "Relaxed Threshold" means moving the target AWAY from the minimum.
    # If min is 0, and f_best is 10, move target to 11?
    # If min is -3.8, and f_best is -3.8, move target to -3.4?
    
    # The hint says "90% of f_best".
    # If f_best = 10, 0.9 * 10 = 9. This is CLOSER to 0 (the min). This is STRICTER.
    # If f_best = -10, 0.9 * -10 = -9. This is CLOSER to 0 (the min, which is 0 for Ackley? No, Ackley min is 0).
    # For Ackley, min is 0. f_best starts ~10. Target 9 is closer to 0. Stricter.
    # This contradicts "maintaining exploration pressure" if it makes it stricter.
    
    # Unless... the "relaxed threshold" refers to something else.
    # Maybe it means $f_{target} = f_{best} + \xi$?
    # Or maybe the hint implies that for the *specific functions* where this improved, the effect was different?
    # Or maybe I am misinterpreting "90% of f_best".
    # What if it means $f_{target} = 0.9 \cdot f_{best}$ only if $f_{best} < 0$?
    
    # Let's look at the score improvement: 0.8363 -> 1.111.
    # This is a significant jump.
    
    # Let's try a different structural change: **Cost-Adjusted UCB**.
    # UCB is often more robust than EI in low-budget regimes.
    # Score = -mu + beta * sigma - lambda * cost
    # We need to tune beta and lambda.
    # beta can be adaptive based on remaining budget.
    
    # Let's try:
    # 1. Compute z for EI.
    # 2. Compute EI.
    # 3. Compute UCB term.
    # 4. Combine.
    
    # Actually, let's look at the "Relaxed Threshold" again.
    # If the program used "90% of f_best" and it worked, maybe my logic about "stricter/easier" is flawed in context.
    # Or maybe "f_best" in the hint refers to the *distance* from the minimum?
    
    # Let's try a robust **Hybrid**:
    # Score = (EI + Exploration_Bonus) / cost
    # Where Exploration_Bonus is based on uncertainty.
    
    # Let's go with **Cost-Aware UCB** as the structural change.
    # It is a different "design family" (Confidence Bound vs Improvement Expectation).
    
    # Parameters:
    # beta: Exploration coefficient.
    # We don't know the iteration count, but we can estimate it from cost.
    # Initial cost ~ 5. Max cost 30.
    # Let's use a fixed beta that balances exploration.
    # A common choice is beta = 1.0 or 2.0.
    # Since sigma is in objective units, and f_best is in objective units,
    # and cost is ~1-2, we need beta to be comparable to the scale of improvement.
    
    # Let's try:
    # Score = -mu + beta * sigma - cost
    # If beta is too high, we explore too much.
    # If beta is too low, we exploit too much.
    
    # Let's try to adapt beta to the spread of f_best?
    # No, f_best is a scalar.
    
    # Let's try **Probability of Improvement (PoI) / Cost**.
    # PoI is bounded [0,1]. EI is unbounded.
    # PoI might be more stable.
    
    # Let's implement PoI / Cost.
    
    # z = (f_best - mu) / (sigma + 1e-9)
    # po_i = norm.cdf(z)
    # score = po_i / cost
    
    # This is a strong candidate. It avoids the "small EI" problem.
    # If mu is slightly better than f_best, z is large positive, PoI ~ 1.
    # If mu is much worse, z is large negative, PoI ~ 0.
    # It rewards points that are *likely* to be better.
    
    # Let's try this.
    
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    # Clip z to avoid numerical issues in cdf
    z = np.clip(z, -10, 10)
    po_i = norm.cdf(z)
    
    scores = po_i / (cost + eps)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
