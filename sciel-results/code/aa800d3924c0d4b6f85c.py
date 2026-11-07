import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization on Rosenbrock 2D.
    
    Uses a hybrid of Probability of Improvement (POI) and cost-aware exploration.
    POI is preferred over EI for escaping the narrow Rosenbrock valley.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Safety: clamp to positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-score: how many standard deviations below f_best
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using normal CDF approximation
    # For numerical stability, use the standard normal CDF
    # P(Z < z) where Z ~ N(0,1)
    # We can use the error function: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    poi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    
    # Also compute an exploration bonus based on sigma
    # We want to explore points with high uncertainty, but penalize high cost
    # Use a cost-aware UCB component
    beta = 0.5
    ucb_component = (f_best - mu + beta * sigma) / cost
    
    # Combine POI and UCB
    # POI is in [0, 1], ucb_component can be negative or positive
    # Normalize ucb_component to be comparable
    # Actually, let's use a weighted sum
    # POI is bounded, so let's scale it up
    poi_weight = 1.0
    ucb_weight = 0.5
    
    scores = poi_weight * poi + ucb_weight * ucb_component
    
    # Alternative: use POI with cost penalty directly
    # POI already favors points likely to improve, but doesn't account for cost
    # Let's create a cost-aware POI
    # The expected number of improvements per unit cost
    # This is tricky. Let's try: POI * (1 / cost)
    # But this might be too aggressive on cost
    
    # Let's try a different approach: 
    # Use POI as the main signal, but add a small exploration term
    # that is cost-aware
    
    # Final score: POI + small exploration bonus
    # Exploration bonus: sigma / cost (encourages high uncertainty, low cost)
    exploration_bonus = 0.1 * sigma / cost
    
    scores = poi + exploration_bonus
    
    # Handle non-finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        # Return immediately if taking too long
        return scores
    
    return scores
