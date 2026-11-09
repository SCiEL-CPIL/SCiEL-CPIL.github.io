import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma_safe = np.maximum(sigma, 1e-8)
    # Cost-dependent xi: higher cost -> higher xi (more conservative)
    # cost is in [1, 2]. Let xi range from 0 to 0.5
    xi = 0.5 * (cost - 1.0)
    
    f_best_adj = f_best - xi
    z = (f_best_adj - mu) / sigma_safe
    ei = (f_best_adj - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # We still divide by cost to account for the budget consumption?
    # The task says "Cost-aware acquisition".
    # Usually, cost-aware means we trade off improvement vs cost.
    # Dividing by cost is a simple way.
    # Using xi is another way.
    # Let's combine?
    # If we use xi, we are already penalizing high cost by demanding more improvement.
    # Dividing by cost again might be too harsh.
    # Let's try just EI_xi without dividing by cost, or with a mild division.
    
    # Let's try EI_xi / cost**0.5
    return ei / np.sqrt(cost)
