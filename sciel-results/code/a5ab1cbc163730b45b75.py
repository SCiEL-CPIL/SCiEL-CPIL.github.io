"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f.
    # LCB value is mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB value.
    # However, the evaluator picks the HIGHEST score.
    # So we return the negative of the LCB value: -mu + beta * sigma.
    # This makes high scores correspond to low mu (good) and high sigma (exploration).
    # 
    # For Michalewicz 2D, we want aggressive exploration early on to find all basins.
    # Using a slightly higher beta helps sweep the multimodal landscape.
    # The function is smooth with known minimum at -1.8013, so we can be confident
    # in the GP predictions after a few evaluations.
    beta = 3.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
