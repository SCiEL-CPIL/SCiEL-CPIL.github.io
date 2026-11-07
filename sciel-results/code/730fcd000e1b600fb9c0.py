import numpy as np

def acquisition(mu, sigma, f_best, cost):
        """
        Cost-aware UCB acquisition function for minimization.
        """
        # UCB for minimization: we want to minimize f(x).
        # Lower bound on f(x) is mu - beta * sigma.
        # We want to maximize the probability of finding a low value.
        # So we score by -mu + beta * sigma.
        # Divide by cost to account for budget.
        
        beta = 2.0
        
        # Calculate UCB score
        ucb = -mu + beta * sigma
        
        # Cost-aware score
        # cost is in [1, 2]
        scores = ucb / cost
        
        # Ensure finite output
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        return scores
