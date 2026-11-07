"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    UCB(x) = mu(x) + k * sigma(x)
    
    Since we are minimizing, we want to find points with low values.
    However, standard UCB is often formulated for maximization.
    For minimization, we can maximize:
    -mu(x) + k * sigma(x)  (if we treat -f as the objective to maximize)
    OR
    We can simply look for the maximum of the "upper bound" of the function value?
    
    Let's stick to the standard minimization formulation.
    We want to select the point that minimizes the expected value but with an exploration bonus.
    Usually, for minimization, we define the acquisition function as:
    alpha(x) = -mu(x) + k * sigma(x)
    
    Wait, let's think about the standard UCB for minimization.
    If we want to minimize f(x), we are interested in the lower bound of f(x).
    The lower bound is mu(x) - k * sigma(x).
    We want to maximize the probability of finding a small value.
    Alternatively, we can view this as maximizing the "regret" potential?
    
    Actually, a common approach for minimization is to use:
    UCB_min(x) = mu(x) - k * sigma(x)
    And we pick the x that MINIMIZES this value? No, that would be pure exploitation if k=0.
    If k is large, we pick high sigma (exploration).
    But `acquisition` returns scores where "higher meaning evaluate next".
    
    So if we use `score = -mu(x) + k * sigma(x)`, then:
    - Low mu (good value) -> high score (because of -mu).
    - High sigma (uncertainty) -> high score (because of +k*sigma).
    This correctly balances finding good values and exploring uncertain regions.
    
    Let's use k = 1.0 as a starting point, which is a common default.
    """
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB for minimization:
    # We want to maximize the score.
    # A low mu is good. A high sigma is good for exploration.
    # score = -mu + k * sigma
    k = 1.0
    scores = -mu + k * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
