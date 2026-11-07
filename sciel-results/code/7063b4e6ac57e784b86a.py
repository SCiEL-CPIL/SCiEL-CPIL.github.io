"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, keep 5s margin
    if time.time() - _start_time > 325:
        # Fallback to simple EI if we run out of time
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    # Cost is 1 + mean(x), so in [1, 2].
    # Higher cost = worse. We want to minimize cost for a given improvement.
    # Let's use a batch size K=5 to encourage diverse but high-value points.
    K = min(5, n)
    
    # 1. Initial scores: Cost-aware EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Cost-aware EI = EI / cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    scores = ei / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # 2. Greedy batch selection
    # We want to return scores for all candidates that reflect their value 
    # in a batch of size K. 
    # A common approximation: 
    # Pick top K by initial score.
    # For the remaining candidates, their "marginal" value is reduced 
    # because we already picked K good ones.
    # However, the loop evaluates ONE point per iteration.
    # The evaluator picks the argmax of the returned scores.
    # So we just need to return scores that make the single best choice 
    # good, while penalizing redundancy if the "best" is too close to 
    # previously evaluated points? No, we don't have coordinates.
    
    # Wait, the evaluator picks the BEST one from the 2000 scores returned.
    # It does NOT pick a batch. It picks x* = argmax(acquisition).
    # So "batch-aware" in the sense of returning a batch is not how the loop works.
    # The loop is:
    # 1. Compute scores for 2000 candidates.
    # 2. Pick x_best = argmax(scores).
    # 3. Evaluate x_best.
    # 4. Update GP.
    
    # So the "batch" strategy doesn't apply directly to the return interface 
    # unless we try to influence the *sequence* of picks.
    # But we can't. We just pick one.
    
    # Alternative Strategy: **Lookahead / Diversity-aware EI**
    # Since we can't pick a batch, we can penalize points that are "similar" 
    # to points that are also high-scoring, to encourage exploration of 
    # different regions if the current best is local.
    # But without coordinates, how do we know similarity?
    # We don't. We only have mu, sigma, cost.
    # However, mu and sigma are derived from the GP which is spatial.
    # High mu and low sigma usually means "known good".
    # Low mu and high sigma means "unknown, potentially good".
    
    # The current strategy is CEI (Cost-EI).
    # What is structurally different?
    # **Knowledge Gradient (KG)** approximation.
    # KG asks: "What is the expected value of knowing the minimum over a set of points?"
    # Approximate KG for a single point x:
    # E[ max(0, f_best - min(f(x), f_best)) ]
    # This is similar to EI but uses the distribution of the minimum.
    # For a single point, min(f(x), f_best) where f(x) ~ N(mu, sigma^2).
    # Let Y = f(x). We want E[ max(0, f_best - min(Y, f_best)) ].
    # If Y < f_best, improvement is f_best - Y.
    # If Y >= f_best, improvement is 0.
    # So it's exactly EI.
    # So for a single point, KG = EI.
    
    # What about **Probability of Improvement (PI)**?
    # P(f(x) < f_best) = Phi(z).
    # Cost-aware PI = Phi(z) / cost.
    # This is more exploratory than EI because it ignores the magnitude of improvement.
    
    # What about **Upper Confidence Bound (UCB)**?
    # UCB = mu - beta * sigma (since we minimize).
    # Or mu + beta * sigma? No, minimize -> mu - beta * sigma.
    # Cost-aware UCB = (mu - beta * sigma) / cost.
    # This is very different from EI. EI is non-linear in z. UCB is linear.
    # UCB can be tuned with beta.
    
    # Let's try **Cost-aware UCB**.
    # The challenge is choosing beta.
    # A standard choice is beta = log(n * t) or similar.
    # Here n is number of evaluated points. We don't have n directly, 
    # but we can infer it from cost?
    # Cost accumulates. Initial 5 points.
    # We don't have access to the count of evaluations.
    # We can track it in a global variable.
    
    global _eval_count
    if '_eval_count' not in globals():
        _eval_count = 0
    _eval_count += 1
    
    # Heuristic beta: start high, decay?
    # Or use a fixed beta that balances exploration/exploitation.
    # For Hartmann 3D, a moderate beta might work.
    # Let's try beta = 2.0 (in standard deviation units).
    # UCB = mu - 2.0 * sigma
    # Score = (mu - 2.0 * sigma) / cost
    
    # But wait, mu is in objective units. f_best is around -3.
    # mu - 2*sigma might be very low (good) if sigma is large.
    # This encourages high uncertainty regions.
    
    # Let's combine EI and UCB?
    # Or just switch to UCB.
    
    # Another option: **Expected Improvement with a penalty for high cost**
    # The current one divides by cost.
    # Maybe subtract a penalty?
    # Score = EI - lambda * cost.
    # If EI is small, cost dominates.
    
    # Let's stick to **Cost-aware UCB** as a structural change from EI.
    # EI is "greedy" on expected gain. UCB is "pessimistic" on the lower bound.
    
    beta = 2.0
    ucb = mu - beta * sigma
    scores = ucb / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
