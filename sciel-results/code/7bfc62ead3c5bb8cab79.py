import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound for minimization
    # We want to maximize the lower bound of the function value: mu - beta * sigma
    # So we return (beta * sigma - mu) / cost
    # Higher score means more promising (lower predicted value relative to uncertainty and cost)
    beta = 3.0
    scores = (beta * sigma - mu) / cost
    return scores
