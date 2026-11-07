import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process posterior, shape (N,).
    sigma (np.ndarray): Standard deviation of the Gaussian Process posterior, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores are preferred.
    """
    # Ensure sigma is strictly positive to avoid any numerical edge cases
    sigma = np.maximum(sigma, 1e-8)
    
    # Exploration parameter for UCB
    # Tuned for the small budget (30 evals) of the Goldstein-Price problem.
    beta = 2.5
    
    # For minimization, we want to maximize the score.
    # -mu rewards low predicted function values (exploitation).
    # beta * sigma rewards high uncertainty (exploration).
    scores = -mu + beta * sigma
    
    return scores
