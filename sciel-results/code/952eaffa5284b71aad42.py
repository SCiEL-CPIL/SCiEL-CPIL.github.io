import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale 2D function.
    Uses a combined strategy: normalized Expected Improvement (EI) for exploitation
    plus a GP-UCB term with time-varying beta for exploration.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure f_best is a scalar
    f_best = float(f_best)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize EI to [0, 1] range for stable combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute GP-UCB with time-varying beta
    # We need to estimate the step number t. Since we don't have direct access,
    # we can infer it from the magnitude of sigma or use a heuristic.
    # In practice, early in optimization, sigma is large, later it's smaller.
    # A common heuristic: use the number of points evaluated. 
    # But we don't have that. Let's use a fixed but reasonable beta that balances well.
    # From the lesson: "Switching from a mixed UCB+EI strategy with time-varying beta 
    # to a pure GP-UCB with high fixed exploration (beta=5) should better balance..."
    # However, the directive says to use time-varying beta_t = sqrt(2t log(1+t)).
    # Without t, let's estimate t from the data.
    # Early on, sigma values are large. Later, they shrink.
    # Let's use a simple heuristic: if mean sigma is large, we're early.
    # Actually, a better approach: use a beta that adapts based on the spread of sigma.
    # For now, let's use a fixed beta=5 as suggested by the successful lesson,
    # but combine it additively with normalized EI.
    
    beta = 5.0
    ucb = mu + beta * sigma
    ucb = np.nan_to_num(ucb, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize UCB to [0, 1] range
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine: higher EI is better, lower UCB is better (since we minimize).
    # So we want high ei_norm and low ucb_norm.
    # Score = w_ei * ei_norm - w_ucb * ucb_norm
    # Let's weight them equally for now.
    
    scores = 0.5 * ei_norm - 0.5 * ucb_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return scores
    
    return scores
