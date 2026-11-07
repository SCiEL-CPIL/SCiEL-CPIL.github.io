import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Cost-Weighted Upper Confidence Bound (UCB) for minimization
        # Score = (f_best - mu + beta * sigma) / cost
        # Higher score is better.
        
        beta = 2.0
        
        # Calculate optimistic potential
        # We want to find points where mu is low and sigma is high
        # f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
        
        score = (f_best - mu + beta * sigma) / cost
        
        # Handle any non-finite values (should not happen with valid inputs)
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
