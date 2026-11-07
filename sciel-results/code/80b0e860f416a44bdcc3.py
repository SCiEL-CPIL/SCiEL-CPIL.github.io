import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Upper Confidence Bound
    # This structural change shifts from Expected Improvement (exploitative)
    # to UCB (optimistic/explorative), which is often better for small budgets
    # where the GP might be poorly calibrated or under-explored.
    beta = 5.0
    ucb = mu + beta * sigma
    scores = ucb / cost
    return scores
