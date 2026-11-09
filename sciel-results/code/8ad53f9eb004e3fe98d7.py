"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0

def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) as the base score, but applies a
    diversity penalty to candidates that are close to other high-scoring
    candidates. This encourages the algorithm to explore multiple
    promising regions in parallel, which helps with multimodal functions.
    
    Since coordinates are not passed, we approximate diversity using
    the GP mean and variance structure. Candidates with similar (mu, sigma)
    profiles are considered "close" in the latent space. We use a simple
    clustering-like penalty: for each candidate, we subtract a penalty
    proportional to the number of other candidates with similar (mu, sigma)
    and high EI scores.
    
    To keep computation efficient (O(N^2) is too slow for N=2000 in Python loops,
    but vectorized O(N^2) might be okay if optimized, or we can use a simpler
    heuristic), we use a different approach:
    
    We rank candidates by raw EI. Then we penalize candidates based on their
    rank proximity to the top-ranked candidates in the (mu, sigma) space.
    
    Actually, a simpler and more robust strategy for multimodal functions
    when coordinates are not available is to use **Thompson Sampling** or
    **UCB** with a stronger exploration term, or to use a **multi-start**
    heuristic embedded in the score.
    
    However, since we must return a single score per candidate, and the
    evaluator picks the single best, we need to make the best candidate
    likely to be in a *different* basin than previously explored ones.
    
    A common trick is to use **Expected Improvement with a penalty for
    high confidence** (i.e., low sigma) if the point is not significantly
    better than f_best. This encourages exploring high-uncertainty areas.
    
    Let's try a **UCB (Upper Confidence Bound)** style score:
    UCB = mu + kappa * sigma
    But since we are minimizing, we want to find low values.
    So we can use:
    Score = -mu + kappa * sigma
    This balances finding low mean values with high uncertainty.
    
    However, EI is generally better for minimization.
    
    Let's look at the "diversity" idea again.
    If we assume the GP is well-calibrated, regions with high sigma are
    likely unexplored. Greedy EI tends to pick points with high EI, which
    often have moderate sigma.
    
    A different strategy: **Probability of Improvement (PoI) with a
    heavy exploration bonus**.
    
    Let's try a **modified EI** that strongly rewards high sigma:
    EI_mod = EI + lambda * sigma
    
    Where lambda is a large constant. This makes the acquisition function
    behave more like UCB, pushing exploration into unexplored areas.
    
    Another idea: **Batch-aware scoring via local density**.
    We can compute a simple "density" of high-EI points in the (mu, sigma)
    space and penalize points in dense regions.
    
    Given the constraints (no coordinates, 2000 points), let's implement
    a **UCB-style acquisition** which is known to be more exploratory than EI
    and can help with multimodality.
    
    UCB for minimization:
    We want to minimize f.
    Lower bound on f is mu - kappa * sigma.
    We want to maximize the potential for improvement.
    
    Let's use a score that combines EI and sigma:
    score = EI / cost + beta * sigma / cost
    
    We will increase beta significantly compared to the current program.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Base score: EI normalized by cost
    score_ei = ei / cost
    
    # Exploration score: sigma normalized by cost
    score_sigma = sigma_safe / cost
    
    # Combine: Heavy weight on exploration to escape local optima
    # The current program uses a small uncertainty bonus.
    # We will use a much larger weight on sigma to force exploration.
    # This effectively turns it into a UCB-like method.
    
    # Adaptive beta: increase beta when EI is low (stagnation)
    # If EI is high, we can afford to exploit more.
    # If EI is low, we need to explore more.
    
    ei_max = np.max(ei)
    if ei_max < 1e-6:
        beta = 10.0  # Very high exploration if no improvement expected
    else:
        # Scale beta inversely with EI magnitude
        # If EI is large, beta is small (exploit)
        # If EI is small, beta is large (explore)
        beta = 1.0 / (1.0 + ei_max)
        beta = np.clip(beta, 0.1, 5.0)
        
    # Final score
    score = score_ei + beta * score_sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
