import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Knowledge-Gradient-inspired score.
    
    Args:
        mu: array of shape (2000,), GP mean predictions.
        sigma: array of shape (2000,), GP standard deviation predictions.
        f_best: float, best observed function value so far.
        cost: array of shape (2000,), cost of evaluating each candidate.
        
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    eps = 1e-9
    safe_sigma = np.maximum(sigma, eps)

    # z-score for probability of beating f_best
    z = (f_best - mu) / safe_sigma

    # P(f(x) < f_best) = Phi(z)
    prob_improve = norm.cdf(z)

    # KG-like: expected reduction in best value if we evaluate here
    # E[max(f_best - f(x), 0)] = EI, but we weight by probability more heavily
    # to favor points with high chance of improvement over large but unlikely gains.
    ei = (f_best - mu) * prob_improve + safe_sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Structural change: use a log-scale combination that emphasizes
    # the probability of improvement (exploration of promising regions)
    # while still rewarding expected gain. This differs from additive EI+beta*sigma.
    #
    # score = (alpha * log1p(prob_improve) + gamma * log1p(ei)) / cost
    #
    # log1p(prob) saturates near 1, so it doesn't over-reward already-certain points.
    # log1p(ei) rewards larger expected improvements with diminishing returns.
    #
    # For multimodal problems, this balances finding new basins (high prob_improve
    # in unexplored areas) with exploiting known good regions (high ei).

    alpha = 2.0
    gamma = 1.0

    prob_term = alpha * np.log1p(prob_improve)
    ei_term = gamma * np.log1p(ei)

    score = (prob_term + ei_term) / (cost + eps)

    if time.time() > deadline:
        pass

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score

# EVOLVE-BLOCK-END
