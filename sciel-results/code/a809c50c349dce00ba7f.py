"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior predictive
    # at each candidate point, then return negative sample as score.
    # Lower predicted value => higher score (since we minimize).
    #
    # z ~ N(0, 1) is a single scalar, so the sample is:
    #   sample = mu - sigma * z
    # and we return -sample = -mu + sigma * z
    #
    # This naturally balances exploration (sigma * z term) and exploitation
    # (-mu term) without a manually tuned beta parameter.
    #
    # We use a fixed z value derived from a deterministic hash-like seed
    # to keep results reproducible per-call while still providing stochasticity
    # across different calls (since the evaluator calls this 25 times per seed,
    # and we want each call to potentially explore differently).
    #
    # Actually, for Thompson Sampling in a loop, we want a fresh random draw
    # each time. But the function is called 25 times (30 evals - 5 initial),
    # and we need finite scores. Let's use np.random with a seed that changes
    # per call to ensure diversity, but we don't have a call counter.
    #
    # Alternative: use a fixed z that provides good exploration. Since we can't
    # track state, let's use a deterministic but varied approach:
    # Use z = 0.5 (a moderate exploration level). This is equivalent to
    # sampling from the 69th percentile of the posterior.
    #
    # Actually, the simplest correct Thompson Sampling: draw z ~ N(0,1) once.
    # The randomness comes from the random number generator state.
    # Since we can't control it, let's just use np.random.randn().
    
    # Draw a single standard normal sample
    z = np.random.randn()
    
    # Score: higher means evaluate next. For minimization, we want low values.
    # Thompson sample = mu - sigma * z (for minimization, lower is better)
    # We return negative of the sample so that higher score = lower predicted value
    score = -mu + sigma * z
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
