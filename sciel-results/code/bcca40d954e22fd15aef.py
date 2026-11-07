import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB: mu - beta * sigma
    # We want to minimize, so lower mu is better.
    # High sigma is good for exploration.
    # Score = (mu - beta * sigma) / cost
    # Higher score is better.
    # So we want high (mu - beta * sigma)?
    # Wait.
    # If we minimize f(x), we want low f(x).
    # UCB for minimization: u(x) = mu(x) - beta * sigma(x).
    # We want to maximize u(x)?
    # No, if mu is low, u is low.
    # We want to find the point with the lowest potential value.
    # So we should MINIMIZE u(x)?
    # Or MAXIMIZE -u(x)?
    # Let's check.
    # If mu = 1, sigma = 0, u = 1.
    # If mu = 2, sigma = 0, u = 2.
    # We prefer mu=1.
    # So we want to minimize u.
    # But the task says "Return 2000 finite scores (higher = evaluate next)".
    # So we should return -u(x) / cost?
    # Or ( -mu + beta * sigma ) / cost?
    
    # Let's re-verify EI logic.
    # EI = (f_best - mu) Phi(z) + sigma phi(z).
    # If mu < f_best, EI is positive.
    # We maximize EI.
    # So higher EI is better.
    
    # For UCB:
    # We want to minimize f(x).
    # The upper bound on the minimum is roughly mu - beta sigma?
    # No, the upper bound on the value is mu + beta sigma.
    # The lower bound is mu - beta sigma.
    # We are looking for the global minimum.
    # We want to pick x that minimizes the lower bound?
    # No, we want to pick x that has a chance of being very low.
    # So we want to minimize mu - beta sigma?
    # If we minimize mu - beta sigma, we prefer low mu and high sigma.
    # This is correct for exploration.
    # So the "UCB value" to minimize is $L(x) = \mu(x) - \beta \sigma(x)$.
    # Since we need to return a score where HIGHER is better,
    # we should return $-L(x) / \text{Cost}(x)$.
    # i.e., $(\beta \sigma - \mu) / \text{Cost}$.
    
    beta = 2.0
    score = (beta * sigma - mu) / cost
    return score
