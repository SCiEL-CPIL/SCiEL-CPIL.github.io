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
    beta = 2.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
