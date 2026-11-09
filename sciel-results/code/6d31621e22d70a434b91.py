# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    modest Upper Confidence Bound (UCB) term for balancing exploitation
    and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Posterior mean at candidate points.
    sigma : array-like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term with a moderate beta
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB. 
    # We want higher scores to mean "evaluate next".
    # EI is already "higher is better" (improvement over f_best).
    # UCB is "lower is better" for minimization, so we negate it or use -ucb.
    # However, mixing EI (which can be negative) and -UCB requires care.
    # A common approach is to simply use EI, or use a weighted sum.
    # Since the task says "higher meaning evaluate next", and EI is standard,
    # let's stick with a robust variant of EI. 
    # The insight mentioned reducing UCB beta and weighting it lower.
    # Let's try a combination: score = ei - alpha * ucb? No, that's not right.
    # Standard UCB for minimization is: score = -mu + beta * sigma.
    # Let's try a hybrid: primarily EI, but with a small exploration bonus
    # that encourages points with high uncertainty if EI is not overwhelmingly positive.
    
    # Alternative: Use a "Probability of Improvement" (PI) or "Thompson Sampling" like score?
    # No, the interface returns scores for 2000 points, and the evaluator picks the max.
    # So we need a deterministic score.
    
    # Let's use a weighted sum of EI and a UCB-like term.
    # UCB score for minimization: -mu + beta * sigma
    # EI score: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # To ensure exploration, we can add a small multiple of sigma to EI?
    # Or use a "Lookahead" style?
    # Given the insight: "Reducing UCB beta from 4 to 1.5 and weighting it lower relative to EI"
    # This implies a combination like: score = w1 * EI + w2 * (-mu + beta * sigma)
    # But EI already contains -mu.
    
    # Let's try a simpler, robust approach: 
    # Score = EI + gamma * sigma
    # This adds a small exploration bonus.
    gamma = 0.1
    
    scores = ei + gamma * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
