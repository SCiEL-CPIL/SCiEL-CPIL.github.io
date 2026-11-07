import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Robust sigma
        sigma = np.maximum(sigma, 1e-6)
        
        # EI
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # UCB (for minimization, lower is better, so we use -mu + beta*sigma as a "desirability" score?)
        # Actually, UCB value is mu - beta*sigma. We want to minimize this.
        # So desirability is -(mu - beta*sigma) = -mu + beta*sigma.
        beta = 1.0
        ucb_desirability = -mu + beta * sigma
        
        # Combine
        # Normalize? They are in different units (EI is improvement, UCB is value).
        # Let's just add a small multiple of UCB desirability to EI.
        # Or use UCB only?
        
        # Let's try pure UCB with cost.
        # Score = - (mu - beta * sigma) / cost
        
        # Let's try a hybrid.
        # Score = (EI + 0.1 * sigma) / cost
        
        return ei / cost
