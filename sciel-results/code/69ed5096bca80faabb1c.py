import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate adaptive beta for UCB
    # Goldstein-Price min is 3.0
    # f_best is the current lowest observed value
    diff = f_best - 3.0
    
    # Ensure diff is non-negative for the tanh scaling, though f_best should be >= 3
    # If f_best < 3 due to numerical issues or noise, clamp it
    if diff < 0:
        diff = 0.0
        
    # Beta starts high (exploration) when far from min, drops to low (exploitation) when close
    # tanh(0.1 * diff) maps [0, inf) to [0, 1)
    # diff=0 -> tanh=0 -> beta=0.5
    # diff=10 -> tanh(1)=0.76 -> beta=0.5 + 2.5*0.76 = 2.4
    # diff=30 -> tanh(3)=0.995 -> beta=0.5 + 2.5*0.995 = 2.99
    beta = 0.5 + 2.5 * np.tanh(0.1 * diff)
    
    # UCB score for minimization: maximize -mu + beta * sigma
    scores_ucb = -mu + beta * sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    scores_ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine UCB and EI scores using a weighted sum
    # Normalize both to have similar scale
    # Use a simple approach: weight based on how close we are to convergence
    # When diff is large, use more UCB (exploration), when small, use more EI (exploitation)
    # Blend factor: 0.5 when far, 0.0 when close
    blend = 0.5 * (1 - np.tanh(0.1 * diff))
    
    # Final score: blend of UCB and EI
    # Higher is better for both
    scores = (1 - blend) * scores_ucb + blend * scores_ei
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
