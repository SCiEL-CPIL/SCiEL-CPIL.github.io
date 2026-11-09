"""Thompson Sampling + cost-aware acquisition for Hartmann-6d BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Pre-compute a random seed for Thompson sampling to be deterministic across calls
# within a single acquisition evaluation but allow variation between evaluations
# by using a global counter or just a fixed seed for reproducibility.
# Actually, Thompson sampling should use a random draw each time the function is called
# to provide stochastic exploration. However, since the function is called once per
# evaluation step (scoring 2000 candidates), we can use a single random draw for all
# 2000 candidates in that step.

# To ensure we don't hit the time limit, we add a simple check.
# The task says: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
# However, the `acquisition` function is called repeatedly in a loop. It doesn't know the global start time directly unless we pass it or use a global.
# The prompt says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores... Each evaluation must finish within 300 s."
# This usually implies the *entire* run of 300s is for the whole loop of evaluations. But the function itself is just scoring.
# The "deadline guard" likely refers to the main loop in the evaluator, but we are only writing `acquisition`.
# Wait, the prompt says: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
# This is tricky for a stateless function. However, we can assume the "evaluation" mentioned in "Each evaluation must finish within 300 s" refers to the entire optimization run for one instance?
# No, "The evaluator measures the raw value... The search score is seed/value".
# "Each evaluation must finish within 300 s." likely means the whole script execution for one instance.
# Since `acquisition` is just a helper, we can't easily track global time unless we use a global variable.
# Let's assume the standard interpretation: the function should be fast.
# However, the directive says "track elapsed time". I will implement a global timer that resets when the process starts (or is assumed to be fresh) and if we are close to 300s, we fall back to a simpler, faster heuristic or just return the current best guess quickly.
# But `acquisition` doesn't know when the run started.
# Let's look at the contract again: "While total cost is below 30, the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
# The 300s limit is for the whole run.
# I will add a global timestamp to track the start of the process.

_start_time = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 5.0  # seconds

def _time_remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - (time.time() - _start_time)

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling score:
    Draw y ~ N(mu, sigma^2) for all candidates.
    Score = -y / cost (since we want to minimize y, higher score is better).
    Actually, the prompt says "selecting the point that minimizes y divided by the cost".
    So score = -y / cost.
    
    We also add a small deterministic exploitation term to ensure we don't ignore
    known good points entirely if sigma is very small, or to guide towards the
    minimum more aggressively.
    """
    # Check time
    if _time_remaining() < 0:
        # Fallback: return a simple deterministic score based on mu only
        # This is a safety net to avoid timeout
        return -mu / cost

    # Thompson Sampling
    # Draw a single random sample for all candidates to maintain correlation?
    # No, typically you draw independent samples for each candidate.
    # But for efficiency and to mimic a single posterior sample, we can draw
    # a standard normal vector and scale it.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw samples
    # y = mu + sigma * z
    z = np.random.randn(len(mu))
    y = mu + sigma * z
    
    # Score: we want to minimize y/cost, so we return -y/cost
    # To avoid division by zero or negative costs (cost is always > 0 here),
    # cost is 1 + mean(x) which is in [1, 2].
    scores = -y / cost
    
    # Add a small exploration bonus for points with high uncertainty?
    # Thompson sampling already does this.
    # However, pure TS can be too noisy.
    # Let's blend with a deterministic EI-like term for stability.
    # But the directive says "Make a STRUCTURAL change... Thompson Sampling".
    # Let's stick to pure TS but ensure numerical stability.
    
    # One issue: TS can pick points with very high cost if y is very low (negative).
    # The cost normalization handles this.
    
    # Return finite scores
    # Replace any non-finite values (shouldn't happen, but just in case)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
