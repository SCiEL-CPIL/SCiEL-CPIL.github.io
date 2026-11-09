import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Cost-Adjusted Probability of Improvement
        # Dynamic scaling: As f_best improves, we demand a stricter improvement.
        # We approximate the "remaining budget" effect by scaling the threshold.
        # Since we don't have total_cost, we use a fixed aggressive gamma.
        # gamma = 0.05 is a reasonable start for Hartmann 6d (range ~ 3.3)
        
        # To make it "dynamic" without loop state, we can scale the threshold by the current best value's magnitude?
        # Or just use a fixed shift.
        
        # Let's use a shift that is proportional to the standard deviation of the best point?
        # We don't have sigma for the best point.
        
        # Let's try: target = f_best - 0.01
        # This forces the POI to be calculated against a slightly worse target, effectively demanding more improvement.
        
        target = f_best - 0.01
        z = (target - mu) / (sigma + 1e-9)
        poi = norm.cdf(z)
        
        # Cost adjustment
        # To prevent division by zero or very small costs, cost is >= 1.
        score = poi / cost
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
