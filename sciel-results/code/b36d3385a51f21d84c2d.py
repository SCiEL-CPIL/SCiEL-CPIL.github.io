"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware LCB: balance exploration/exploitation while preferring low-cost points.
    # beta controls the exploration-exploitation trade-off.
    beta = 0.5
    # LCB: lower is better (we want to minimize)
    lcb = mu - beta * sigma
    # Scale by inverse cost: higher score = more desirable
    # Use 1/cost to normalize, and negate LCB so that lower LCB (better) gives higher score
    score = -lcb / (cost + 1e-9)
    return score
# EVOLVE-BLOCK-END
