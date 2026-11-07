import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI)
    with a dynamic improvement threshold xi that scales with f_best.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic improvement threshold
    # xi = 0.1 * f_best
    # This forces the search to target regions with high probability of
    # achieving a significant relative improvement.
    # Early on (f_best ~ 25), xi ~ 2.5, so we need to find points much better than best.
    # Late on (f_best ~ 0.1), xi ~ 0.01, so we just need any improvement.
    # When f_best is very small, xi approaches 0, behaving like standard PI.
    xi = 0.1 * max(f_best, 0.0)
    
    # Calculate z-score for PI
    # PI(x) = P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    # Wait, for minimization:
    # We want P(f(x) < f_best - xi)
    # Let Y ~ N(mu, sigma^2). We want P(Y < f_best - xi).
    # Z = (Y - mu) / sigma ~ N(0, 1)
    # P(Y < f_best - xi) = P(Z < (f_best - xi - mu) / sigma) = Phi((f_best - xi - mu) / sigma)
    
    z = (f_best - xi - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima, especially early on.
    # Use a small coefficient to not overwhelm the PI signal.
    # The bonus is proportional to sigma, but scaled down.
    # As f_best decreases, we want less exploration.
    # Let's use a bonus that is active when f_best is large.
    # bonus = 0.1 * (f_best / 25.0) * sigma_safe
    # But PI already decreases as f_best decreases (xi decreases), so maybe just a small constant bonus?
    # Or no bonus, let PI do the work?
    # Let's try a small bonus to ensure we don't get stuck if PI becomes too greedy.
    # bonus = 0.05 * sigma_safe * (f_best / 25.0)
    
    # Actually, PI can be very peaked. Adding a small UCB-like term might help.
    # But the directive says "different design or modelling family".
    # PI is a different family from EI.
    # Let's stick to pure PI first, or PI with a very small exploration term.
    
    # Let's try PI + small exploration term
    # exploration_weight = 0.1 * (f_best / 25.0)
    # scores = pi + exploration_weight * sigma_safe
    
    # Let's try just PI first.
    scores = pi
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
         return scores
         
    return scores
