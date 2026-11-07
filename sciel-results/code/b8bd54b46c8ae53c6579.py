import numpy as np

def acquisition(mu, sigma, f_best):
    # Base exploration rate based on current uncertainty
    base_kappa = np.mean(sigma)
    
    # Scale exploration based on how far we are from the optimum
    # If f_best is high (bad), explore more. If f_best is low (good), exploit more.
    # Ackley min is 0.
    # Use a soft scaling.
    # If f_best is 0, factor is 1. If f_best is 10, factor is 11.
    # This might be too aggressive.
    # Let's use exp or log.
    # factor = 1.0 + f_best
    
    # Simpler: just use mean(sigma).
    # The "Dynamic" part is that kappa changes as the GP updates (sigma changes).
    
    score = -mu + base_kappa * sigma
    return score
