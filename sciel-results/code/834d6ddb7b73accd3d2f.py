"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy: Variance-Regularized Lower Confidence Bound (VR-LCB) with Quadratic Cost Penalty.
    
    Unlike Expected Improvement which focuses on the probability/expectation of improving f_best,
    this approach uses an optimistic bound on the minimum value:
    LCB(x) = mu(x) - beta * sigma(x)
    
    To handle the cost-awareness (minimizing 1 + mean(x)), we subtract a quadratic cost penalty.
    This encourages the search to find good values in low-cost regions (low coordinates)
    while still exploring high-uncertainty areas, but penalizing high-cost exploration more
    heavily than a linear penalty would.
    
    Score = LCB(x) - lambda * cost(x)^2
    
    We also normalize sigma to ensure stability and use a fixed beta tuned for the Hartmann 6D
    problem scale. The quadratic cost penalty (power 2) aligns with the finding that stronger
    cost penalties help align the search with the low-coordinate global optimum.
    """
    t_start = time.time()
    
    # Ensure inputs are safe
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Hyperparameters
    # Beta controls the exploration-exploitation trade-off.
    # In Hartmann 6D, the function is relatively smooth but multi-modal.
    # A beta around 2.0-3.0 is often effective for 6D problems with Matern-5/2.
    # We use 2.5 as a strong exploration factor.
    beta = 2.5
    
    # Lambda controls the strength of the cost penalty.
    # Cost is in range [1, 2] (since x in [0,1]^6, mean(x) in [0,1], cost = 1+mean(x)).
    # A penalty of ~1.0-2.0 can significantly shift preference to low-cost regions.
    # Based on the "Stronger cost penalty (power 2)" insight, we use a moderate-to-strong penalty.
    lambda_cost = 1.5
    
    # Calculate LCB
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Calculate Cost Penalty
    # Quadratic penalty to strongly discourage high-cost points
    cost_penalty = lambda_cost * (cost ** 2)
    
    # Final Score
    # Higher is better for the acquisition function selection
    # Since we are minimizing f, a lower LCB is better.
    # But the acquisition function needs to return scores where HIGHER = evaluate next.
    # So we want to maximize: -f_predicted + exploration - cost_penalty?
    # Wait, standard BO acquisition functions like EI or POI return positive scores for improvement.
    # LCB is usually used to MINIMIZE the LCB value.
    # So, if we return `score = -LCB + ...`? No, that's maximizing -LCB, which is minimizing LCB.
    # Let's stick to the convention: Return a score where higher is better.
    # If we want to minimize f, we want low mu and high sigma (uncertainty).
    # LCB = mu - beta*sigma. We want to MINIMIZE LCB.
    # Therefore, the acquisition score should be proportional to -LCB.
    # Score = -(mu - beta*sigma) - lambda*cost^2 = beta*sigma - mu - lambda*cost^2
    
    score = beta * sigma - mu - cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Time check
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        # If we somehow spent too long (unlikely for vectorized ops), return what we have
        # Replace -inf with a very small number to avoid issues if any exist
        score = np.where(np.isfinite(score), score, -1e10)
        return score
        
    # Replace any remaining -inf with a very small number to ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
