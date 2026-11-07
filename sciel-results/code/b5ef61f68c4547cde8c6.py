import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    epsilon = 1e-9
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # Avoid division by zero
    sigma_safe = sigma + epsilon
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use a stronger exploration strategy: Pure UCB with adaptive beta
    # For minimization, we want to explore areas with high uncertainty
    # and low predicted mean. UCB for minimization: -mu + beta * sigma
    # Use a higher beta to encourage more exploration, especially in early stages
    # when f_best is far from the optimum.
    
    # Adaptive beta: start with high exploration, reduce as f_best improves
    # Since we don't know the true minimum (0), use f_best as a proxy.
    # If f_best is large, we're far from optimum -> explore more.
    # If f_best is small, we're close -> exploit more.
    # Scale beta between 0.5 (exploit) and 3.0 (explore) based on f_best.
    # Ackley's maximum value is around 29.8, minimum is 0.
    # Use a smooth transition: beta = 0.5 + 2.5 * sigmoid(-f_best)
    # But sigmoid might be too slow. Let's use a simpler decay.
    # beta = max(0.5, 3.0 - f_best)
    # Or even simpler: just use a fixed high beta since the problem is 2D and small.
    # Let's try a fixed beta of 2.0, which is more exploratory than 1.5.
    
    beta = 2.0
    ucb_score = -mu + beta * sigma
    
    # Blend EI and UCB
    # Increase the weight of UCB to encourage more exploration
    weight_ei = 0.5
    weight_ucb = 0.5
    blended_score = weight_ei * ei + weight_ucb * ucb_score
    
    # Apply cost penalty
    # Use a softer cost penalty to allow evaluating slightly more expensive points
    # if they have high acquisition potential.
    cost_safe = cost + epsilon
    penalty = np.power(cost_safe, 0.3)
    
    # Final acquisition score
    scores = blended_score / penalty
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
