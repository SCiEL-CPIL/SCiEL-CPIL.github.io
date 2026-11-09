import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Cost-aware Lower Confidence Bound (LCB) for minimization
        # Equivalent to maximizing: -mu + beta * sigma / cost
        
        beta = 2.0
        
        # Ensure cost is safe for division
        # Cost is 1 + mean(x_scaled), so cost >= 1.
        # But just in case, clamp to a small positive number.
        safe_cost = np.where(cost > 1e-8, cost, 1.0)
        
        # Calculate score
        # -mu: Exploitation (lower mu is better, so -mu is higher)
        # beta * sigma / cost: Exploration (higher sigma is better, lower cost is better)
        score = -mu + (beta * sigma) / safe_cost
        
        return score
