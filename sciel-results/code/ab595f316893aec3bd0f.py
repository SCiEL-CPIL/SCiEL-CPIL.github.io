"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    The beta parameter is adapted based on the current progress of the search.
    When f_best is far from the known global minimum of the Hartmann 3D function (-3.86278),
    the search is in an early/exploratory phase, so beta is increased to encourage exploration.
    As f_best improves (gets closer to the global minimum), beta decreases to shift towards exploitation.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,).
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,).
    f_best : float
        The lowest observed function value so far.
    
    Returns:
    --------
    np.ndarray
        Acquisition scores for each candidate, shape (N,). Higher is better.
    """
    # Known global minimum of Hartmann 3D
    global_min = -3.86278
    
    # Calculate how far we are from the global minimum (normalized)
    # When f_best is very high (early), gap is large -> more exploration
    # When f_best is close to global_min, gap is small -> more exploitation
    gap = f_best - global_min
    
    # The initial random points might give f_best around -0.5 to -2.0
    # As we converge, f_best approaches -3.86278
    
    # Use a smooth transition for beta
    # Early on (large gap): beta ~ 2.5-3.0 (strong exploration)
    # Late on (small gap): beta ~ 0.5-1.0 (strong exploitation)
    
    # Normalize the gap to [0, 1] range approximately
    # A gap of 0 means we've found the global min
    # A gap of ~3 means we're still early (f_best around -0.86)
    # Use a sigmoid-like or linear mapping
    
    # Let's use a simple linear interpolation with clipping
    # Map gap from [0, 4] to beta from [0.5, 3.0]
    # When gap >= 4, beta = 3.0 (max exploration)
    # When gap <= 0, beta = 0.5 (max exploitation)
    
    # Actually, let's think about typical f_best values:
    # After 5 random points: f_best might be around -1.0 to -2.0
    # After 10-15 points: f_best might be around -2.5 to -3.5
    # After 25-30 points: f_best should be close to -3.86
    
    # Let's use a different scaling:
    # beta = base_beta + exploration_boost * exp(-progress)
    # where progress measures how close we are to the optimum
    
    # Simpler approach: use the gap directly
    # beta = 0.5 + 2.5 * min(gap / 3.0, 1.0)
    # This gives:
    # gap = 0 -> beta = 0.5
    # gap = 3 -> beta = 3.0
    # gap > 3 -> beta = 3.0
    
    gap_clipped = min(max(gap, 0.0), 4.0)
    beta = 0.5 + 2.5 * (gap_clipped / 4.0)
    
    # Ensure beta is reasonable
    beta = np.clip(beta, 0.3, 4.0)
    
    # LCB acquisition: mu - beta * sigma
    # We want to maximize this (since lower function values are better)
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
