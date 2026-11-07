"""Rewritten program using UCB with decaying exploration for better multimodal search."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of mean values from the GP posterior, shape (2000,).
        sigma: Array of standard deviation values from the GP posterior, shape (2000,).
        f_best: The best (lowest) objective value observed so far.
        
    Returns:
        Array of scores, shape (2000,). Higher scores indicate better candidates.
    """
    # Time step estimation: We don't have t directly, but we can infer it from f_best?
    # No, we don't have access to the history length. 
    # However, the problem states the loop runs for 30 evaluations.
    # We can't easily get 't' inside the function without global state.
    # Let's use a fixed beta or a beta based on the spread of sigma?
    # Actually, we can just use a fixed high beta. 
    # If we want to be adaptive without 't', we can use the magnitude of sigma.
    # But a fixed beta is safer.
    
    # Let's use a fixed beta of 5.0. This is a strong exploration factor.
    # Score = mu + beta * sigma
    # We want to MINIMIZE f, so we want LOW mu.
    # UCB for minimization is usually: mu - beta * sigma? 
    # No, UCB is for MAXIMIZATION: mu + beta * sigma.
    # For MINIMIZATION, we want to find low values.
    # The "Upper Confidence Bound" on the *true* function value (which we suspect might be low) 
    # is actually the Lower Confidence Bound (LCB) if we are minimizing.
    # LCB = mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB.
    # So the SCORE should be high for LOW LCB.
    # Score = -LCB = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's verify:
    # If mu is low (good) and sigma is high (uncertain), -mu is high (good) and beta*sigma is high (good).
    # So Score = -mu + beta * sigma.
    
    beta = 5.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
