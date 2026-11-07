import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) / Upper Confidence Bound (UCB) logic.
    For minimization, we want to find points with low predicted mean (mu) and high uncertainty (sigma).
    The score is defined as (beta * sigma - mu) / cost, where beta adapts to the current best value.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Dynamic beta: 
    # When f_best is high (far from optimum 0), we want more exploration (higher beta).
    # When f_best is low (close to optimum), we want more exploitation (lower beta).
    # f_best for Ackley is in [0, 4].
    # beta ranges from ~1.0 (exploit) to ~9.0 (explore).
    beta = 1.0 + 2.0 * f_best
    
    # UCB score for minimization:
    # We maximize this score to select the next point.
    # A high score indicates either a very low mu (good prediction) or high sigma (uncertainty).
    ucb_score = beta * sigma_safe - mu
    
    # Cost-aware adjustment:
    # Divide by cost to prefer cheaper evaluations.
    # cost is in [1, 2].
    scores = ucb_score / cost
    
    # Ensure all scores are finite
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
