"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, leave margin before 330s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware EI with exponential cost penalty and progress-adaptive lambda.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _remaining_time() < 0.5:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, -1e10)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Progress-adaptive lambda: as f_best gets closer to 0 (better),
    # lambda increases, making the cost penalty stronger.
    # When f_best is large (early), lambda is small, allowing exploration.
    # When f_best is near 0 (late), lambda is large, strongly penalizing cost.
    # Scale: f_best for rosenbrock can range from ~0 to ~100+
    # lambda = 1 / (1 + f_best) gives lambda in (0, 1]
    # We want stronger differentiation, so scale up:
    progress = max(f_best, 1e-6)
    lambda_base = 1.0 / (1.0 + progress)
    # Amplify the effect: early lambda ~ 1/(1+large) ~ small, late ~ 1/(1+small) ~ large
    # But we want the penalty to be more aggressive in late stages
    # Use: lambda = c / (1 + f_best) where c controls overall strength
    # At f_best=0: lambda = c, at f_best=10: lambda = c/11
    # Let's use c=5.0 for a strong late-stage penalty
    c = 5.0
    lam = c / (1.0 + progress)
    
    # Exponential cost penalty: exp(-lambda * cost)
    # Cost ranges from 1 to 2 (since cost = 1 + mean(x_scaled_to_[0,1]))
    # So lambda * cost ranges from lambda to 2*lambda
    # For lambda=5 (f_best near 0): penalty ranges from exp(-5)~0.0067 to exp(-10)~4.5e-5
    # For lambda=0.45 (f_best~10): penalty ranges from exp(-0.45)~0.64 to exp(-0.9)~0.41
    # This creates a smooth, strong cost barrier in late stages
    
    # Numerical safety for exp
    penalty_arg = -lam * cost_safe
    penalty_arg = np.maximum(penalty_arg, -50.0)  # avoid underflow
    penalty = np.exp(penalty_arg)
    
    # Score: EI * cost_penalty
    # This strictly favors low-cost candidates, with the strength of preference
    # adapting to optimization progress
    score = ei * penalty
    
    # Add a small exploration bonus to prevent premature convergence
    # The bonus is proportional to sigma, scaled down as we progress
    exploration_weight = 0.1 / (1.0 + progress)
    exploration_bonus = exploration_weight * sigma_safe
    
    score = score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
