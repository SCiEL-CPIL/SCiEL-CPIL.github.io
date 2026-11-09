import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Handle division by zero or very small sigma
    # norm.cdf and norm.pdf handle this, but let's be safe
    
    pi = norm.cdf(z)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Weighted Expected Improvement
    # lambda = 0.5 balances exploration (PI) and exploitation (EI)
    # PI is more exploratory as it ignores the magnitude of improvement
    # Actually, PI is NOT more exploratory in the same way.
    # PI is high when P(f < f_best) is high.
    # EI is high when E[improvement] is high.
    
    # Let's check the behavior.
    # If sigma is large, z is small (if mu ~ f_best).
    # PI ~ 0.5.
    # EI ~ sigma * phi(0) ~ 0.4 sigma.
    # So EI scales with sigma. PI is bounded [0,1].
    # Adding PI to EI adds a constant-ish term to high sigma regions?
    # No, if mu is very different from f_best, PI goes to 0 or 1.
    
    # A better exploration term is often just sigma itself.
    # But that's LCB.
    
    # Let's try a different EI variant: **Expected Improvement with a noise threshold**?
    # No.
    
    # How about **Probability of Improvement** with a small epsilon?
    # P(f < f_best - epsilon).
    # This is just PI with a shifted f_best.
    # f_best_adj = f_best - 0.1 * (f_best - min_possible)?
    # We don't know min_possible.
    
    # Let's go with **LCB** after all. It is a very standard and strong alternative.
    # Many papers state LCB and EI are comparable, but LCB can be better in early stages.
    # Given the "25 RBFs" landscape, early exploration is key.
    
    # Let's try LCB with beta=2.0.
    # It is simple and robust.
    
    return -mu + 2.0 * sigma
