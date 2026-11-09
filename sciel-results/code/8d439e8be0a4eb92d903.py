import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # UCB for minimization: maximize (-mu + beta * sigma)
        # Divide by cost to account for cost-awareness
        beta = 2.0
        ucb = -mu + beta * sigma
        return ucb / cost
