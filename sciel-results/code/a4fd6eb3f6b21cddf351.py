# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    We generate a single random sample from the posterior predictive distribution
    f(x) ~ N(mu(x), sigma(x)^2) and return these samples as the acquisition scores.
    
    For minimization, we want to find points with low function values.
    Since the evaluator picks the candidate with the HIGHEST score to evaluate next,
    we return -sample (negative of the sampled function value).
    This way, points where we sampled a very low function value get a high score (less negative),
    and points where we sampled a high function value get a low score (more negative).
    
    Wait, let's re-read the contract:
    "evaluates the candidate with the highest score"
    "Scoring (lower is better): mean over the seeds of the simple regret"
    "mu and sigma ... in objective units"
    "Return 2000 finite scores, higher meaning evaluate next"
    
    So if we sample y ~ N(mu, sigma^2), and we want to minimize the objective,
    we should prefer points with low y.
    If we return y as the score, the highest score corresponds to the highest y, which is bad.
    So we should return -y. Then the highest score corresponds to the lowest y.
    
    However, Thompson Sampling typically works by minimizing the sampled function.
    If the external loop does `x_next = x_candidates[np.argmax(scores)]`,
    and we want `x_next` to be where the sampled function is low,
    we must return `scores = -sampled_function`.
    
    Let's verify:
    If sampled_function is -3.8 (good), score = 3.8.
    If sampled_function is -2.0 (bad), score = 2.0.
    argmax picks 3.8, which corresponds to -3.8. Correct.
    
    This is a stochastic acquisition function. It naturally balances exploration and exploitation
    because points with high uncertainty (high sigma) have a wider range of possible sampled values,
    increasing the chance that a very low value is sampled, thus getting a high score.
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu
    
    # Handle near-zero sigma to avoid issues, though not strictly necessary for sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a single random sample from the posterior predictive distribution
    # f(x) ~ N(mu, sigma^2)
    # We use standard normal random numbers scaled by sigma and shifted by mu
    random_samples = np.random.randn(mu.shape[0])
    sampled_function = mu + sigma_safe * random_samples
    
    # For minimization, we want to maximize the score.
    # The score should reflect "how good is this point likely to be?"
    # In Thompson Sampling for minimization, we pick the point with the minimum sampled value.
    # Since the evaluator picks the MAXIMUM score, we invert the sampled values.
    score = -sampled_function
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
