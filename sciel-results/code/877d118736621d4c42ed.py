"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    # New Strategy: Cost-Aware Expected Improvement with Uncertainty Bonus
    # Instead of UCB (which can be too aggressive or too passive depending on beta),
    # we use a hybrid that combines:
    # 1. Expected Improvement (EI) - drives exploitation
    # 2. Uncertainty (sigma) - drives exploration
    # 3. Cost penalty - penalizes expensive points
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Hybrid score: EI + alpha * sigma - beta * cost
    # The alpha * sigma term encourages exploring high-uncertainty regions
    # The beta * cost term penalizes expensive evaluations
    # This is different from pure EI/cost or pure UCB/cost
    
    # We want to balance exploitation (EI) with exploration (sigma)
    # and cost efficiency (cost)
    
    # Let's try: score = (ei + 0.5 * sigma) / cost
    # This adds a small exploration bonus to EI, then normalizes by cost
    
    # Alternatively, we can use a weighted combination:
    # score = (0.8 * ei + 0.2 * sigma) / cost
    
    # Let's use a slightly more aggressive exploration term:
    # score = (ei + 0.3 * sigma) / cost
    
    # But let's make it more principled. Let's use:
    # score = (ei + lambda * sigma) / (1 + cost)
    # where lambda controls the exploration-exploitation tradeoff
    
    # Actually, let's try a different structural approach:
    # Use **Expected Improvement** but with a **Cost-Adjusted Z-score**
    
    # Or even better: **Cost-Weighted Probability of Improvement**
    # POI = Phi(z)
    # score = POI * (f_best - mu) / cost
    # This is similar to EI but without the PDF term, making it more robust
    
    # Let's try: score = (f_best - mu) * norm.cdf(z) / cost
    # This is the "Expected Improvement" without the exploration term from sigma*phi(z)
    # It's more conservative but might be more stable
    
    # Actually, let's go with a **Hybrid EI-UCB** with cost awareness:
    # UCB_value = mu - beta * sigma (for minimization, we want low values)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Let's combine them:
    # score = (0.7 * ei + 0.3 * (f_best - mu + 1.0 * sigma)) / cost
    
    # This blends exploitation (EI) with optimism (UCB-like term)
    
    # Let's implement this:
    ucb_term = f_best - mu + 1.0 * sigma
    score = (0.7 * ei + 0.3 * ucb_term) / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
