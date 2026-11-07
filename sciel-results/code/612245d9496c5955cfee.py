import numpy as np

def acquisition(mu, sigma, f_best, cost):
            # Cost-aware UCB for minimization
            # We want to maximize the score.
            # Score = (f_best - mu + beta * sigma) / cost
            beta = 1.5
            # Ensure cost is not zero (though it should be >= 1)
            cost_safe = np.maximum(cost, 1e-9)
            scores = (f_best - mu + beta * sigma) / cost_safe
            # Ensure finite
            return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
