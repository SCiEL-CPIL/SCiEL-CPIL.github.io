"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    # If we are close to the limit, return a neutral score immediately
    if time.time() - _START_TIME > 310:
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Base Score: Cost-Aware EI
    # We use EI / cost as a strong baseline proxy.
    base_score = ei / cost
    
    # Two-Stage Strategy:
    # Stage 1: Select top K candidates based on base_score.
    # Stage 2: Perform a local refinement around these candidates to find a better point.
    # Since we only have mu, sigma, cost vectors (no x-coordinates), we cannot
    # directly optimize x. However, we can assume that the "best" candidate
    # in the batch is a good proxy for the local optimum.
    
    # Wait, the problem states: "The evaluator scores 2000 new uniform candidates 
    # with acquisition and evaluates the best one."
    # This implies we return scores for 2000 points. The evaluator picks argmax(score).
    # If we just return base_score, the evaluator picks the best of 2000 uniform points.
    
    # How can we do "nested search" if we don't have x-coordinates?
    # We can't move x. We can only score the given 2000 points.
    # However, we CAN use the structure of the 2000 points to improve the score.
    
    # Alternative Interpretation of "Different Strategy":
    # The current program uses UCB: (f_best - mu + 1.5*sigma) / cost.
    # The inspiration 1 (score 1.92) used a hybrid EI + Uncertainty.
    # The current program is UCB.
    # Let's try a **Knowledge Gradient (KG) Approximation** or **Expected Improvement with a stronger exploration term**.
    
    # Actually, looking at the inspirations:
    # Inspiration 1 (1.92): "Cost-Aware Expected Improvement with Uncertainty Bonus"
    #   score = (EI + alpha * sigma) / cost ? Or similar.
    # Inspiration 2 (1.80): "Two-stage nested acquisition" - but how?
    #   Maybe it uses the fact that we can pre-compute something?
    
    # Let's look at the constraint: "No coordinates are passed."
    # This means we cannot do local optimization in x-space.
    # The "Two-stage" in Inspiration 2 might refer to something else, or it failed/was not fully described.
    
    # Let's try a **Bayesian Optimization with Cost-Weighted UCB** but with a **Dynamic Beta**.
    # Or better: **Expected Improvement (EI)** is generally superior to UCB for minimization 
    # when the goal is simple regret, because UCB can be too exploratory.
    # The current program uses UCB.
    # Let's switch to **EI** with a **Cost Penalty** that is **Sub-linear**.
    
    # Strategy: **Cost-Aware EI with Log-Cost Penalty**
    # score = EI - lambda * log(cost)
    # Or **EI / (cost ** 0.5)**
    
    # Let's test **EI / (cost ** 0.5)**.
    # This penalizes cost less than linear division, allowing slightly more expensive points 
    # if their EI is significantly higher.
    
    # Let's also consider **Probability of Improvement (POI)**.
    # POI = Phi(z).
    # POI is more robust to outliers in sigma than EI.
    
    # Let's try a **Hybrid**:
    # score = (0.7 * EI + 0.3 * POI * (f_best - mu)) / (cost ** 0.5)
    
    # Actually, the most distinct and robust improvement over UCB is often **EI**.
    # Let's implement **EI / (cost ** 0.5)**.
    
    # Calculate POI just in case
    poi = norm.cdf(z)
    
    # Combine EI and POI?
    # EI is the expected reduction. POI is the probability of reduction.
    # EI is usually better for minimizing regret.
    
    # Let's try **EI / (cost ** 0.5)**.
    scores = ei / (cost ** 0.5)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
