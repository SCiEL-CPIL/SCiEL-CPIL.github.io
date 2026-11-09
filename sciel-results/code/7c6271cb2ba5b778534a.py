"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track evaluated points for diversity penalty
# We infer evaluated points by looking at where sigma is minimal (high certainty)
# and where the GP mean matches the observed f_best or other low values.
# However, a simpler and more robust heuristic for "evaluated points" in a black-box
# setting where we only get mu/sigma at 2000 random points is to use the
# "expected improvement" baseline but add a repulsive term based on the
# density of points with low sigma (high confidence) in the vicinity.

# To strictly adhere to the "no coordinates passed" constraint, we must infer
# the location of past evaluations from the posterior.
# Points that have been evaluated will have very low sigma (close to noise floor).
# We can identify "anchors" as points in the 2000 candidates where sigma is
# extremely low, and penalize new candidates that are close to these anchors.
# But wait, we don't have the coordinates of the 2000 candidates.
# This is a critical constraint: "No coordinates are passed."
# Therefore, we CANNOT compute distances to past points directly.

# Re-evaluating the strategy:
# If we cannot use coordinates, we cannot do explicit distance-based diversity.
# We must rely on the GP posterior itself.
# A structural change from EI (which exploits local uncertainty) to
# "Thompson Sampling" style or "Probability of Improvement" is still local.
#
# Let's look at the inspirations again.
# Inspiration 1: EHVI (batch-aware).
# Inspiration 2: Linear score (f_best - mu) + beta * sigma.
#
# The current program is EI.
# A materially different strategy that doesn't require coordinates:
# **Probability of Improvement (PI) with a tuned constant** or **UCB**.
# But UCB is mentioned in inspiration 2/3.
#
# Let's try a **Hybrid Acquisition** that blends UCB and EI, or specifically
# a **Lookahead-free, uncertainty-weighted exploitation**.
#
# Actually, the most distinct structural change available without coordinates
# is to change the *objective* of the acquisition from "expected reduction in
# f" to "probability of being the global minimum" or similar.
#
# However, a very effective and distinct strategy for rugged landscapes like
# Ackley is **Expected Improvement with a restart mechanism** or **Multi-Start
# EI**. But we can't restart.
#
# Let's consider **Knowledge Gradient (KG)** approximation.
# KG is structurally different from EI. It estimates the expected improvement
# in the *knowledge* of the optimal value, not the immediate value.
# Approximation: KG ≈ EI + (sigma^2 / (f_best - mu)) * something?
# A common simple approximation for KG in 1D is:
# KG ≈ (f_best - mu) * Phi(z) + sigma * phi(z) + (sigma^2 / (f_best - mu)) * phi(z) * ...
#
# Let's try **UCB (Upper Confidence Bound)** as the primary structural change.
# EI is greedy and can get stuck. UCB is explicitly optimistic about uncertainty.
# The current program is EI. The inspirations suggest UCB worked well (1.538).
# But the directive says "COMPUTE BUDGET... STAGNATION... STRUCTURAL change".
# Switching from EI to UCB is a standard swap, but is it "materially different"?
# Yes, it changes the trade-off from expected value to worst-case regret bound.
#
# Let's implement a **tuned UCB** with an adaptive beta that scales with the
# number of evaluations (if we could track it) or just a fixed, well-chosen beta.
# Since we don't track eval count, we use a fixed beta.
#
# Wait, the "inspiration 2" score was 1.0933 for a linear score.
# The "inspiration 1" score was 1.5379 for EHVI.
# The current program (EI) is 1.0.
#
# Let's try a **Weighted Sum of UCB and PI**.
# Or better, let's try **Thompson Sampling** approximation?
# No, TS requires sampling from the posterior, which is hard without coordinates
# to define the covariance structure for the sample. We only have pointwise mu/sigma.
#
# Let's go with **UCB** as the structural change. It is fundamentally different
# from EI in that it does not integrate over the distribution but uses a
# quantile-like bound.
#
# Formula: UCB = mu + beta * sigma
# For minimization, we want to pick points that might be low.
# So we minimize the lower bound: LCB = mu - beta * sigma.
# Higher score = better. So Score = -LCB = -mu + beta * sigma.
#
# Let's refine: The task is minimization.
# EI: E[f_best - f(x) | f(x) < f_best].
# UCB: We want to find low values. The lower confidence bound is mu - beta*sigma.
# We want to maximize the chance of finding a low value, so we pick the x that
# maximizes the "optimistic" low value? No, we pick the x that minimizes the
# "pessimistic" high value?
#
# Standard UCB for maximization: mu + beta*sigma.
# Standard UCB for minimization: mu - beta*sigma (pick the smallest).
# Since the function returns scores where "higher meaning evaluate next",
# we should return - (mu - beta * sigma) = -mu + beta * sigma.
#
# Let's add a deadline guard as requested.

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin for 330s limit

def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition function for minimization.
    Returns scores where higher is better.
    """
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Return a safe fallback (e.g., pure exploration or exploitation)
        # If we are out of time, just return the most exploitative score
        # to ensure we return something finite.
        return -mu + 1e-9 * sigma

    # UCB for minimization:
    # We want to select points with low mu and high sigma (uncertainty).
    # The lower bound of the function value is mu - beta * sigma.
    # We want to MINIMIZE this lower bound.
    # So the "goodness" score is - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Beta selection:
    # A typical value for 2D Ackley with 30 evals.
    # Too high: pure exploration. Too low: pure exploitation (greedy).
    # EI is effectively adaptive beta.
    # Let's use a fixed beta that balances exploration.
    # For 30 evaluations in 2D, beta=1.0 to 2.0 is often good.
    # Let's try beta=1.5.
    
    beta = 1.5
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
