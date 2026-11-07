import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses adaptive UCB (Upper Confidence Bound) for minimization.
    """
    # f_best is the current minimum observed value.
    # We want to balance exploitation (low mu) and exploration (high sigma).
    # Score to maximize: kappa * sigma - mu
    
    # Adapt kappa based on f_best.
    # When f_best is high (early in optimization), we want more exploration (higher kappa).
    # When f_best is low (late in optimization), we want more exploitation (lower kappa).
    
    # Map f_best to a normalized value in [0, 1].
    # f_best starts potentially around 10-25 and goes to 0.
    # Using f_best / (f_best + 5) provides a smooth saturation.
    f_norm = f_best / (f_best + 5.0)
    
    # Kappa ranges from 0.5 (pure exploitation bias) to 2.5 (strong exploration bias).
    kappa = 0.5 + 2.0 * f_norm
    
    # Calculate UCB score for minimization.
    # We maximize this score.
    score = kappa * sigma - mu
    
    # Handle any numerical issues (though unlikely with GP outputs)
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
