"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) approximation for Bayesian optimization.
    
    The exact KG is:
      KG(x) = E[ max(f_best, min(f_best, f(x))) ]
    For a normal distribution N(mu, sigma^2), this equals:
      KG(x) = mu * Phi((f_best - mu)/sigma) + f_best * (1 - Phi((f_best - mu)/sigma)) - sigma * pdf((f_best - mu)/sigma)
    
    Actually, the exact closed-form for the expected minimum of N(mu,sigma^2) and f_best is:
      E[min(f_best, Y)] where Y ~ N(mu, sigma^2)
      = f_best * P(Y > f_best) + E[Y | Y < f_best] * P(Y < f_best)
      = f_best * (1 - Phi(z)) + (mu - sigma * pdf(z)/Phi(z)) * Phi(z)
      where z = (f_best - mu)/sigma
    
    So:
      = f_best * (1 - Phi(z)) + mu * Phi(z) - sigma * pdf(z)
      = f_best - (f_best - mu) * Phi(z) - sigma * pdf(z)
    
    The improvement in expected best is:
      KG(x) = f_best - E[min(f_best, Y)]
            = f_best - [f_best - (f_best - mu)*Phi(z) - sigma*pdf(z)]
            = (f_best - mu)*Phi(z) + sigma*pdf(z)
    
    Wait, that's exactly EI! Let me re-derive.
    
    Actually, EI is defined as E[max(0, f_best - Y)] which for Y~N(mu,sigma^2) gives:
      EI = (f_best - mu)*Phi(z) + sigma*pdf(z)
    
    And the expected minimum is E[min(f_best, Y)] = f_best - EI.
    
    So KG for a single point IS EI. The difference with true KG is that true KG considers
    the expected value of the global minimum after observing ALL points, not just one.
    
    However, we can implement a better approximation by considering that we have 2000 candidate
    points and computing a sample-based KG. But that would be too expensive.
    
    Alternative: use a hybrid approach that combines EI with a diversity/uncertainty bonus,
    or implement an approximate KG that accounts for the fact that we're choosing one point
    from a set of 2000.
    
    For now, let's implement EI with a small exploration bonus based on sigma to encourage
    exploration of high-uncertainty regions, especially in early stages.
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small uncertainty-based exploration bonus
    # This helps escape local exploitation and encourages exploring high-sigma regions
    # Scale the bonus by a factor that decreases as sigma grows to avoid over-exploration
    exploration_bonus = 0.1 * sigma / (1.0 + sigma)
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
