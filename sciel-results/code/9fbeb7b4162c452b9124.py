import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian Process Lower Confidence Bound (GP-LCB).
    
    Args:
        mu: Array of GP mean predictions for 2000 candidates.
        sigma: Array of GP standard deviations for 2000 candidates.
        f_best: The best observed objective value so far.
        
    Returns:
        Array of scores for the 2000 candidates. Higher scores indicate 
        more promising candidates for the next evaluation.
    """
    # Fixed beta for LCB. 
    # The directive suggests adaptive beta = 1 + 0.1 * t.
    # Without access to t, we use a value that balances exploration and exploitation 
    # well for the 6D Hartmann function with a small budget.
    # t ranges from ~5 to 30. Average t ~ 17.5 -> beta ~ 2.75.
    # We use 3.0 to slightly favor exploration, which is critical for multimodal landscapes.
    beta = 3.0
    
    # LCB for minimization: mu - beta * sigma
    # We maximize this score to select the next point.
    scores = mu - beta * sigma
    
    return scores
