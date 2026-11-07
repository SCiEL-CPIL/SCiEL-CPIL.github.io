"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Structurally different strategy: Diversity-Promoting Probability of Improvement.
    
    Instead of purely pointwise EI or UCB, this approach:
    1. Computes a Probability of Improvement (PoI) score, which is more stable than EI.
    2. Applies a diversity penalty: candidates close to high-cost regions or 
       (implicitly) to previously evaluated points are penalized. 
       Since we don't have explicit coordinates, we use the GP's predictive variance (sigma) 
       as a proxy for "information gain" or "novelty". 
       However, the directive suggests penalizing candidates close to already-evaluated points. 
       Without coordinates, we can't compute direct distances. 
       
       Alternative interpretation: The GP's sigma is low near evaluated points. 
       High sigma means "far from evaluated points" (in terms of GP uncertainty). 
       Therefore, a pure PoI/cost might already favor exploration if cost is uniform, 
       but cost varies. 
       
       Let's try a **Thompson Sampling** inspired approach or a **Knowledge Gradient** approximation. 
       But KG is hard without coords. 
       
       Let's stick to the directive: "batch-aware, diversity-promoting". 
       Since we return 2000 scores for 2000 candidates, the evaluator picks the BEST one. 
       So "batch-aware" in the sense that we want the *selected* point to be diverse. 
       
       If we simply maximize PoI/cost, we might pick a point that is very close to an existing 
       point but has slightly lower mu. To encourage diversity, we can add a term that 
       rewards high sigma (uncertainty), but PoI already decreases with sigma. 
       
       A common trick is to use **UCB** (which rewards sigma) combined with a cost penalty. 
       The current program uses UCB/cost. 
       
       Let's try **Expected Improvement (EI)** but with a **spatial penalty** approximated by 
       the inverse of the local density of points? No, no coords.
       
       What if we use **PoI** but divide by a "cost-adjusted uncertainty"? 
       
       Actually, the most robust structural change from the current UCB/cost (score 0.106) 
       is to go back to the high-scoring inspiration: **PoI**. 
       The inspirations 1 and 2 scored 1.5426 using PoI. 
       The current program is UCB/cost. 
       
       Let's implement **PoI / cost** but with a structural twist: 
       We will use a **cost-aware threshold**. 
       
       Score = PoI * (1 / cost)
       
       But to make it "diversity-promoting" as per the directive, we can add a small 
       exploration bonus based on sigma, but damped by cost. 
       
       Actually, let's look at the "batch-aware" aspect. 
       If we assume the evaluator picks the top-1, then making the scores of points 
       *close* to the current best *lower* encourages diversity. 
       How to detect "close to current best" without coords? 
       We can't. 
       
       However, we can assume that the GP's mu is low and sigma is low near the best point. 
       PoI is high there. 
       If we want to avoid re-evaluating near the best, we should penalize low sigma? 
       No, low sigma means high confidence. 
       
       Let's try **Thompson Sampling**: 
       Sample a value from N(mu, sigma^2) for each point. 
       Then pick the point with the lowest sampled value, adjusted for cost? 
       Score = -sampled_value / cost. 
       This is a stochastic acquisition. It naturally balances exploration and exploitation 
       and tends to pick diverse points because the samples are noisy. 
       
       Let's implement Thompson Sampling with cost adjustment. 
       Score = - (mu + sigma * z) / cost, where z ~ N(0,1). 
       But we need deterministic scores for the 2000 points to return. 
       We can't return random scores each time if the function is called multiple times? 
       The prompt says "Return 2000 finite scores". It doesn't say they must be deterministic 
       across calls, but usually acquisition functions are deterministic given the inputs. 
       
       If we use random z, the scores change every time. The evaluator calls acquisition, 
       gets scores, picks the max, evaluates. 
       If we use random z, we might pick a point that is not the best expected improvement. 
       But Thompson Sampling is a valid BO strategy. 
       
       However, the "diversity-promoting" directive suggests a deterministic penalty. 
       
       Let's try **Cost-Effective Probability of Improvement (CEPoI)**: 
       Score = PoI / cost. 
       This is structurally different from UCB/cost. 
       
       To add "diversity", we can modify the cost to be "effective cost" that increases 
       if the point is in a region of high density? We don't have density. 
       
       Let's try **PoI** with a **sigma bonus** that is scaled by cost. 
       Score = (PoI + lambda * sigma / cost) 
       This encourages exploring high-uncertainty areas if they are cheap. 
       
       Let's go with **PoI / cost** as the base, which matches the high-scoring inspirations, 
       and add a small exploration term to satisfy the "diversity" directive structurally 
       by rewarding high sigma (uncertainty) which implies distance from data points. 
       
       Score = (PoI + 0.1 * sigma) / cost 
       
       This is a structural change from UCB/cost to a PoI-based hybrid. 
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct and inputs are safe
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: fall back to a fast single-variant score
    if time.time() - _START > _LIMIT:
        # Fallback: PoI / cost
        z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
        poi = norm.cdf(z)
        scores = poi / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Compute Probability of Improvement
    z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
    poi = norm.cdf(z)
    
    # Diversity-promoting term: reward high sigma (uncertainty) 
    # This acts as a proxy for information gain / distance from evaluated points.
    # We scale it by a small factor to not dominate the PoI signal.
    # We also divide by cost to maintain cost-awareness.
    lambda_explore = 0.5
    exploration_bonus = lambda_explore * sigma / cost
    
    # Combine: PoI/cost + exploration_bonus
    # Note: PoI is in [0,1], sigma is in objective units. 
    # To make them comparable, we might need to normalize sigma. 
    # But sigma is already in objective units. 
    # Let's just use the sum. 
    scores = (poi / cost) + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
