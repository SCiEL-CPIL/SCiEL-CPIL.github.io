"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Combines the dynamic confidence-based beta from the current program with
    a normalized Expected Improvement per Cost (EIPC) formulation inspired by
    the "value-for-money" strategy.
    
    1. Computes Expected Improvement (EI) for minimization.
    2. Normalizes EI by the candidate's cost to get a "value" metric.
    3. Applies a dynamic penalty for uncertainty (sigma) weighted by a
       confidence factor derived from f_best (low f_best -> high confidence ->
       high penalty for uncertainty).
    
    This shifts from raw EI subtraction to a ratio-based metric that explicitly
    favors cheap points when the expected gain is similar, while allowing
    expensive points if they offer significantly higher expected improvement.
    """
    # Deadline Guard: Check if we are close to the time limit.
    # If so, return a simple, fast score to allow the loop to finish gracefully.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple Expected Improvement, ignoring cost to save time/logic
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Beta based on f_best
    # We map f_best to a "Confidence" score C in [0, 1].
    # High f_best -> Low Confidence -> Low Beta (Explore, ignore uncertainty penalty more)
    # Low f_best  -> High Confidence -> High Beta (Exploit, penalize uncertainty heavily)
    
    # Heuristic parameters for the sigmoid:
    # Assume f_best starts high (e.g., 50-100) and drops to 0.
    # We want the transition to happen when f_best is reasonably low, say < 10.
    # C = 1 / (1 + exp(a * (f_best - b)))
    # If f_best >> b, C -> 0.
    # If f_best << b, C -> 1.
    
    # Let's set b=5.0, a=0.5.
    # If f_best = 50, exp(0.5*45) is huge, C ~ 0.
    # If f_best = 0, exp(0.5*-5) = exp(-2.5) ~ 0.08, C ~ 0.92.
    # If f_best = 5, C = 0.5.
    
    a_param = 0.5
    b_param = 5.0
    
    # Clamp f_best for numerical stability in exp
    f_best_clamped = np.clip(f_best, -100, 100)
    
    confidence = 1.0 / (1.0 + np.exp(a_param * (f_best_clamped - b_param)))
    
    # Beta ranges from 0.1 (low confidence) to 1.0 (high confidence)
    beta_min = 0.1
    beta_max = 1.0
    beta = beta_min + confidence * (beta_max - beta_min)
    
    # Compute the acquisition score
    # Strategy: EIPC (Expected Improvement per Cost) minus Uncertainty Penalty
    # 
    # Term 1: ei / cost_safe
    #   This normalizes the expected gain by the cost.
    #   A point with EI=1 and Cost=1 scores 1.0.
    #   A point with EI=1 and Cost=2 scores 0.5.
    #   This inherently prefers cheaper points when EI is comparable.
    #
    # Term 2: - beta * sigma_safe
    #   This penalizes uncertainty.
    #   By NOT multiplying by cost here, we avoid double-penalizing expensive
    #   uncertain points (they are already penalized in Term 1).
    #   The penalty is purely for the risk of the prediction being wrong.
    #
    # Alternative considered: ei - beta * sigma * cost (Current Program)
    #   This subtracts a cost-weighted uncertainty.
    #   The EIPC approach is structurally different: it's a ratio, not a sum.
    #   Ratios can be more sensitive to cost variations, which is desirable
    #   for cost-aware optimization.
    
    eipc = ei / cost_safe
    
    # The uncertainty penalty should also be scaled somewhat by the "value" scale.
    # If EI is very small, a large sigma penalty might dominate unnecessarily.
    # However, standard EI already accounts for sigma in the calculation.
    # Subtracting beta*sigma is a common heuristic for "risk aversion".
    
    # Let's refine the penalty to be relative to the potential improvement scale.
    # If f_best is large, we can tolerate more uncertainty.
    # If f_best is small, we need high precision.
    # The 'confidence' variable already captures this.
    
    # To ensure the score is comparable across different f_best scales,
    # we might need to normalize the penalty.
    # But since f_best is fixed per call, the relative ordering is what matters.
    
    score = eipc - beta * sigma_safe
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
