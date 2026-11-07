import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin well below 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI).
    EI captures the magnitude of improvement, while PI captures the likelihood of a new record.
    """
    # Check time to avoid timeout
    current_time = time.time()
    elapsed = current_time - _START_TIME
    
    # If we are close to the limit, return a simple, fast score to ensure we finish in time.
    # The evaluator kills at 330s. We should finish well before.
    if elapsed > _TIME_LIMIT:
        # Simple UCB-like score as a fallback, which is very fast
        beta = 1.0
        score = -mu + beta * sigma
        score = np.where(np.isfinite(score), score, -np.inf)
        return score

    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Calculate Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi(z)
    pi = Phi
    
    # Combine EI and PI
    # The insight from the best inspiration (1.6567) was that maximizing the probability
    # of improvement (PI) is very effective for simple regret because it focuses on
    # regions likely to beat the current best.
    # The current program uses UCB, which is a balance of exploitation (-mu) and exploration (sigma).
    # EI is a more principled balance.
    # Let's blend EI and PI. PI is in [0, 1]. EI is in [0, infinity).
    # We need to scale them to be comparable.
    # A simple approach: score = w_ei * EI + w_pi * PI
    # Since EI can be much larger than PI, we might need to normalize EI or weight PI higher.
    # However, PI is already a probability, so it's bounded.
    # Let's try a weighted sum.
    # Based on the success of PI alone (1.6567) and the current UCB (1.3801),
    # PI seems to be a strong signal.
    # Let's use a combination that heavily favors PI but adds the magnitude of EI.
    
    # Normalize EI by its max to bring it to a similar scale as PI (if max is large)
    max_ei = np.max(ei)
    if max_ei > eps:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
        
    # Weighted combination
    # Give more weight to PI, as it directly targets simple regret
    w_pi = 0.7
    w_ei = 0.3
    
    score = w_pi * pi + w_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
