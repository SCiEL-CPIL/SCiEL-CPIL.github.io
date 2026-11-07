"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking to respect the 330s wall-clock limit
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Thompson Sampling variant with adaptive exploration.
    # We sample a scalar offset from a Gaussian distribution.
    # Mean of offset: f_best
    # Std of offset: scales with max uncertainty (max(sigma))
    # This effectively shifts the objective to maximize (offset - mu) / cost.
    # By sampling offset, we balance exploration (high offset) and exploitation (low offset).
    
    # Calculate max uncertainty
    max_sigma = np.max(sigma_safe)
    
    # Define the standard deviation for the Thompson Sampling offset.
    # We want it to be significant enough to encourage exploration early on,
    # but shrink as uncertainty decreases or f_best improves.
    # A common heuristic is to scale with the current best value relative to the problem scale.
    # However, the directive suggests scaling with max(sigma).
    # Let's use a factor that keeps the offset within a reasonable range.
    # If max_sigma is 0, we fall back to exploitation.
    if max_sigma < 1e-6:
        offset_std = 1e-6
    else:
        # Scale the std by a factor. 
        # If f_best is high, we might want more exploration.
        # Let's use a fixed scaling factor for now, e.g., 0.5 * max_sigma.
        # This means the offset can be f_best +/- 0.5 * max_sigma.
        # This creates a dynamic "target" that fluctuates.
        offset_std = 0.5 * max_sigma
        
    # Sample the offset
    # We use a fixed random seed per call? No, we want stochasticity.
    # But we must ensure reproducibility isn't broken if the evaluator expects determinism?
    # The task says "stochastically balances", implying randomness is allowed/expected.
    # However, standard practice in these evolve loops is often deterministic unless specified.
    # Wait, the directive says "sample a single scalar offset from a Gaussian distribution".
    # This implies randomness.
    # To be safe and reproducible in case of re-runs, we could seed based on f_best and max_sigma?
    # No, let's just use np.random. If the evaluator runs multiple times, it will vary.
    # The score is "mean over the five instances", so variance might be averaged out or penalized.
    # Let's stick to the directive.
    
    offset = np.random.normal(f_best, offset_std)
    
    # Score: -(mu - offset) / cost = (offset - mu) / cost
    # This is equivalent to maximizing (offset - mu) / cost.
    # Since offset is constant for all candidates in this call,
    # this is equivalent to maximizing (-mu + offset) / cost.
    # The offset term is a constant bias for all candidates, so it doesn't change the ranking!
    # Wait. (offset - mu) / cost = offset/cost - mu/cost.
    # Since cost varies per candidate, the offset/cost term DOES vary.
    # Candidates with lower cost will have a higher offset/cost term.
    # So this effectively adds a bonus to cheaper candidates, scaled by the sampled offset.
    # If offset is high, cheap candidates are boosted more.
    # If offset is low (or negative), cheap candidates are boosted less (or penalized less).
    
    score = (offset - mu) / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
