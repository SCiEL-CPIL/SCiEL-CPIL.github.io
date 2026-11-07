import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB normalized by marginal cost.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base cost is 1.0 (the minimum possible cost).
    # The "marginal" cost of evaluating a point x is cost(x) - 1.0.
    # However, we must pay the base cost 1.0 regardless.
    # A more accurate "efficiency" metric for UCB in cost-aware BO is often:
    # (f_best - mu + beta*sigma) / cost
    # But let's try a specific structural variant: 
    # Reward the potential improvement relative to the cost paid ABOVE the minimum.
    # Actually, the standard cost-aware UCB is (f_best - mu + beta*sigma) / cost.
    # The current program uses max(LCB/cost, EI/cost).
    # Let's try a pure UCB with a different beta schedule and no max/EI mix.
    # Or better: Use the "Cost-Aware UCB" defined as:
    # UCB_c(x) = (f_best - mu(x) + beta * sigma(x)) / c(x)
    
    # Let's try a different structural angle: 
    # Instead of dividing by total cost, we can view the problem as 
    # maximizing the expected reduction in regret per unit cost.
    
    # A known effective alternative is the "Knowledge Gradient" or "Expected Hypervolume Improvement" (EHVI) variants, 
    # but for 1D output, UCB is strong. 
    # The current score is 0.71, while a parent was 1.15. 
    # The diff shows the parent had a "portfolio of strategies". 
    # The current code is a "max of LCB and EI/cost".
    
    # Let's try a simpler, more robust UCB with a fixed, aggressive beta early on,
    # but crucially, let's normalize by (cost - 1.0 + epsilon) to heavily penalize expensive points?
    # No, cost is in [1, 2]. Dividing by (cost-1) blows up for cheap points.
    # Let's stick to dividing by cost, but change the exploration term.
    
    # Strategy: Use UCB with a beta that decays as we get more data (implied by f_best approaching min).
    # But we don't have N. We have f_best.
    # Let's try a "Softmax" style blending or just a pure UCB with a different beta.
    
    # Actually, the inspiration 1 (score 1.1571) likely used a specific UCB or EI variant.
    # Let's try a "Cost-Aware EI" but with a "sharpened" improvement term?
    
    # Let's try a completely different acquisition: 
    # Probability of Improvement (PoI) per cost? No, PoI is often too greedy.
    
    # Let's go with a **Thrun-Hutter UCB** but with a **time-adaptive beta** 
    # and **no EI max**. The previous code had `np.maximum(lcb, ei_cost)`.
    # The max operation can be unstable if EI and LCB have different scales.
    # Let's try a weighted sum or just pure UCB.
    
    # Pure UCB:
    # progress = (f_best - min_val) / (max_val - min_val)
    # min_val for cosine8 is -0.8.
    # max_val is unknown but let's assume range is roughly [-0.8, 0.8] or similar.
    # f_best starts around 0.
    
    min_val = -0.8
    # Estimate progress: 1.0 when f_best is high (early), 0.0 when f_best is low (late)
    # We need a reference for "high". Let's use 0.0 as a reasonable upper bound for uniform init in 8D cosine.
    ref_max = 0.0 
    progress = np.clip((f_best - min_val) / (ref_max - min_val), 0.0, 1.0)
    
    # Beta schedule: Start high (explore), end low (exploit)
    # beta = 1.0 + 2.0 * progress  # 3.0 early, 1.0 late
    
    # Let's try a different structural idea: 
    # **Expected Improvement per unit cost** is standard.
    # What if we use **Upper Confidence Bound per unit cost** but with a **dynamic beta** 
    # that depends on the **uncertainty of the GP itself**?
    
    # Let's try a simpler structural change:
    # Use **UCB** with a **fixed beta** but normalize by **cost^2**? 
    # Or normalize by **cost** but add a **penalty for high cost** in the numerator?
    
    # Let's try the **Cost-Aware UCB** with a **beta** that is **inversely proportional to cost**?
    # No, that's circular.
    
    # Let's try a **Hybrid** that uses **EI** for exploitation and **UCB** for exploration, 
    # but weighted by the **probability that the point is better than f_best**.
    
    # Actually, let's look at the "inspiration 1" which scored 1.1571. 
    # The current program is a child of that (or related) and dropped to 0.71.
    # The diff shows the parent had a "portfolio of strategies".
    # The current code is a "max of LCB and EI/cost".
    # Maybe the max is the problem?
    
    # Let's try a **Weighted Sum** instead of Max.
    # score = w1 * (LCB / cost) + w2 * (EI / cost)
    
    # Or, let's try **Pure EI / cost** but with a **modified EI** that accounts for cost?
    # Standard EI assumes we get a sample for free.
    
    # Let's try a **Different Normalization**:
    # Instead of dividing by cost, we subtract a cost penalty from the improvement?
    # Improvement = f_best - mu
    # Net Improvement = Improvement - lambda * (cost - 1)
    # This is a **Lagrangian** approach.
    
    # Let's try **Lagrangian Cost-Aware UCB**:
    # Score = (f_best - mu + beta * sigma) - lambda * cost
    
    # We need to choose lambda.
    # If cost is in [1, 2], and improvement is in [-0.8, 0.8], 
    # lambda should be small.
    
    # Let's try a **Hybrid**:
    # 1. Calculate UCB
    # 2. Calculate EI
    # 3. Combine them using a **soft-max** or **geometric mean**?
    
    # Let's try a **Simpler Structural Change**:
    # Use **UCB** with a **beta** that is **adaptive to the number of evaluations**?
    # We don't have N. But we can estimate N from the total cost?
    # No, we don't have total cost.
    
    # Let's try **UCB** with a **beta** that is **constant** but **higher** than before?
    # The current beta is 0.5 to 2.5.
    # Let's try **beta = 3.0** fixed.
    
    # Or, let's try **EI** with a **xi** parameter?
    # EI(xi) = E[max(f_best - xi - f(x), 0)]
    # This shifts the exploitation.
    
    # Let's try **Cost-Aware EI** with **xi = 0.1 * (f_best - min_val)**?
    
    # Let's go with a **Lagrangian** approach as it is a distinct optimization framing.
    # We want to maximize Improvement - Penalty * Cost.
    # Improvement is approximated by (f_best - mu + beta * sigma).
    # Penalty * Cost is lambda * cost.
    
    # How to choose lambda?
    # We can set lambda such that the penalty is comparable to the expected improvement.
    # Expected improvement is roughly O(0.1).
    # Cost is O(1.5).
    # So lambda should be O(0.01 - 0.1).
    
    # Let's try **lambda = 0.1**.
    
    ucb = f_best - mu + 2.0 * sigma_safe
    penalty = 0.1 * cost_safe
    scores = ucb - penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
