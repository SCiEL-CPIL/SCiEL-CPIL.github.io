import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB (Upper Confidence Bound) approach for minimization
    # We transform the minimization problem to a maximization problem by negating the mean.
    # Maximizing -f is equivalent to minimizing f.
    # UCB for maximization: U = mu_target + beta * sigma_target
    # Here target = -f, so mu_target = -mu, sigma_target = sigma.
    # U = -mu + beta * sigma
    
    # Exploration-exploitation trade-off parameter
    beta = 1.0
    
    # Calculate UCB values
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring
    # We divide by cost to prefer cheaper points with similar acquisition values.
    # Cost is guaranteed to be >= 1.
    scores = ucb / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
