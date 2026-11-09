import numpy as np
import time
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Wall-clock deadline guard
# ---------------------------------------------------------------------------
_T0 = time.time()
_WALL_LIMIT = 330.0
_SAFETY = 30.0
_DEADLINE = _WALL_LIMIT - _SAFETY

# Global step counter for adaptive behavior
_STEP = 0
_TOTAL_OPT_STEPS = 25  # 30 total - 5 initial


def _ei_score(mu, sigma, f_best, xi=0.0):
    """Expected Improvement for minimization. Higher = better."""
    eps = 1e-10
    imp = f_best - xi - mu
    sigma_safe = np.maximum(sigma, eps)
    z = imp / sigma_safe
    # Handle edge cases
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where imp <= 0, EI should be 0
    ei = np.where(imp > 0, ei, 0.0)
    return np.maximum(ei, 0.0)


def _lcb_score(mu, sigma, kappa):
    """Lower Confidence Bound for minimization. Higher = better (i.e., lower predicted value + uncertainty)."""
    # LCB = mu - kappa * sigma (lower bound on the function value)
    # We want to maximize this score, so we return -LCB = -mu + kappa*sigma
    # Wait, that's UCB. For minimization, LCB is mu - kappa*sigma.
    # The point with the lowest LCB is the most promising.
    # To make "higher score = better", we can return -LCB = -mu + kappa*sigma.
    # This is equivalent to UCB in the standard maximization sense.
    # Let's stick to the standard definition: Score = -mu + kappa*sigma
    return -mu + kappa * sigma


def _adaptive_acquisition(mu, sigma, f_best):
    """Adaptive acquisition that shifts from exploration to exploitation."""
    global _STEP
    
    # Adaptive parameters based on step
    progress = _STEP / max(_TOTAL_OPT_STEPS, 1)
    
    # Beta for UCB/LCB: start high (exploration), decrease (exploitation)
    beta = 2.0 * (1.0 - progress) + 0.3
    
    # EI weight: increase over time
    ei_weight = 0.5 + 0.5 * progress
    
    # UCB weight: decrease over time
    ucb_weight = 1.0 - ei_weight
    
    # Compute scores
    ei = _ei_score(mu, sigma, f_best, xi=0.0)
    ucb = _lcb_score(mu, sigma, beta)  # Using LCB logic (same formula as UCB for minimization score)
    
    # Normalize both to [0, 1] for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    
    # Blend
    score = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Add tiny random perturbation to break exact ties (deterministic per call)
    n = len(mu)
    if n > 0:
        idx = np.arange(n)
        perturbation = 1e-8 * np.sin(idx * 0.1 + _STEP * 0.01)
        score = score + perturbation
    
    return score


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores (higher = evaluate next).
    
    Uses adaptive acquisition that blends EI and LCB with a diversity bonus.
    The blend shifts from exploration-heavy to exploitation-heavy over time.
    """
    global _STEP
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return EI scores
        out = _ei_score(mu, sigma, f_best)
        _STEP += 1
        return np.where(np.isfinite(out), out, 0.0)
    
    # Compute adaptive acquisition scores
    out = _adaptive_acquisition(mu, sigma, f_best)
    
    # Ensure finite
    out = np.where(np.isfinite(out), out, 0.0)
    
    # Increment step
    _STEP += 1
    
    return out
