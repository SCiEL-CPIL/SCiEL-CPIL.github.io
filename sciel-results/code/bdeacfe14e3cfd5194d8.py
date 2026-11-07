"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid strategy:
    # When sigma is large (uncertainty is high), we want to explore more, 
    # so we rely more on PI which is bounded and smooth.
    # When sigma is small (uncertainty is low), we want to exploit, 
    # so we rely more on EI which scales with the expected gain.
    
    # Normalize EI to be comparable to PI? 
    # Actually, a common hybrid is simply w * PI + (1-w) * EI.
    # However, EI and PI have different scales. 
    # A more robust approach for this specific "different strategy" requirement 
    # is to use PI as the primary driver but add a small exploitation term 
    # or simply use PI itself as it is a fundamentally different acquisition 
    # function family (probabilistic threshold vs expected gain).
    
    # Let's stick to a pure PI for the "MATERIALLY DIFFERENT" requirement, 
    # as PI is distinct from EI. 
    # PI favors points where the probability of being better than f_best is high.
    # This can lead to more exploration than EI in some regimes.
    
    return pi
# EVOLVE-BLOCK-END
