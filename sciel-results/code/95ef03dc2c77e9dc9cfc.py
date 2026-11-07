"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, which is equivalent to maximizing -LCB(x).
    # Score = beta * sigma(x) - mu(x)
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages exploration of uncertain regions.
    # A lower beta encourages exploitation of known low-value regions.
    # For 2D RBF mixture, moderate exploration is often beneficial.
    beta = 2.0
    
    # Calculate LCB score (higher is better for selection)
    # lcb_score = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Ensure finite values
    lcb_score = np.nan_to_num(lcb_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return lcb_score
# EVOLVE-BLOCK-END
