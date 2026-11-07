import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement (EI).
    
    This function corrects the reference frame to use f_best (current best observed)
    rather than 0, which is crucial for minimization problems where the optimum
    is far from zero (e.g., Shekel 4d min ~ -10.5).
    
    It blends a UCB term (aggressive exploration) with an EI term (probabilistic
    exploitation) and normalizes by the evaluation cost.
    
    Score = (alpha * UCB + (1 - alpha) * EI) / cost
    where:
      UCB = (f_best - mu) + beta * sigma
      EI  = (f_best - mu) * Phi(z) + sigma * phi(z), with z = (f_best - mu) / sigma
    """
    start_time = time.time()
    # Safety margin: 300s limit, aim to return well before. 
    # Vectorized ops on 2000 points are very fast (<1ms), so this is mostly precautionary.
    deadline = start_time + 290

    # Convert to numpy arrays and ensure float64 for precision
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to prevent division by zero or invalid math
    # Sigma should be positive
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    # Cost should be positive
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    # Mu should be finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # f_best is a scalar
    f_best = float(f_best)
    
    # --- Component 1: UCB (Upper Confidence Bound) ---
    # For minimization, we want high scores for low mu and high sigma.
    # The term (f_best - mu) represents the expected improvement over the current best.
    # Adding beta * sigma encourages exploration in uncertain regions.
    beta = 2.0
    ucb = (f_best - mu) + beta * sigma
    
    # --- Component 2: Expected Improvement (EI) ---
    # EI is the expected amount by which f(x) improves upon f_best.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # --- Blend UCB and EI ---
    # The inspiration with score 1.0371 used alpha=0.6 for UCB.
    # UCB provides a strong exploration push, while EI provides calibrated exploitation.
    alpha = 0.6
    combined = alpha * ucb + (1.0 - alpha) * ei
    
    # --- Cost Awareness ---
    # Normalize by cost to prefer cheaper evaluations.
    scores = combined / cost
    
    # --- Final Checks ---
    # Ensure all scores are finite. Replace non-finite with a very low score (-inf)
    # so they are never selected.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline check (unlikely to trigger for 2000 points, but required by contract)
    if time.time() > deadline:
        return scores

    return scores
