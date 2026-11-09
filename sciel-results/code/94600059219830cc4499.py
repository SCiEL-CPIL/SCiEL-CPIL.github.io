"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard: if we've used > 295s of the 330s limit,
    # return a simple fallback immediately.
    elapsed = time.time() - _SESSION_START
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        score = ei / np.maximum(cost, 1e-8)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (for minimization)
    # EI = E[f_best - f | f < f_best] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Lower Confidence Bound (for minimization): LCB = mu - kappa * sigma
    # We want to maximize the "improvement" over LCB, or equivalently minimize LCB
    # The acquisition should favor points with low LCB (high improvement potential)
    # So we use -LCB = -mu + kappa * sigma as a score component
    kappa = 1.5
    lcb_score = -mu + kappa * sigma_safe
    
    # Cost-normalized versions
    ei_cost = ei / cost_safe
    lcb_cost = lcb_score / cost_safe
    
    # Adaptive weighting strategy:
    # When f_best is still high (early in optimization), explore more (LCB weight)
    # When f_best is close to optimum, exploit more (EI weight)
    # Use a smooth transition based on f_best relative to typical rastrigin values
    
    # Rastrigin minimum is 0, typical values in [-5.12, 5.12]^2 are 0-40+
    # f_best starts high and decreases
    # Use a sigmoid to blend: high f_best -> more exploration, low f_best -> more exploitation
    
    # Normalize f_best to [0, 1] range approximately (0 to 40 covers most of the range)
    f_best_norm = np.clip(f_best / 40.0, 0.0, 1.0)
    
    # Weight for exploration increases when f_best is high
    # When f_best_norm is high (early), favor LCB more
    # When f_best_norm is low (late), favor EI more
    
    # Use a smooth blend:
    # alpha_ei: weight for EI (exploitation)
    # alpha_lcb: weight for LCB (exploration)
    # alpha_ei + alpha_lcb = 1
    
    # When f_best_norm is high (early), alpha_lcb is high (more exploration)
    # When f_best_norm is low (late), alpha_ei is high (more exploitation)
    
    # Smooth transition using sigmoid-like function
    # alpha_ei = 1 / (1 + exp(-beta * (f_best_norm - 0.5)))
    # But simpler: linear blend with smooth transition
    
    # Use a tanh-based smooth step for better behavior
    # transition = 0.5 * (1 + tanh(3 * (0.5 - f_best_norm)))
    # This gives ~0 when f_best_norm > 0.5, ~1 when f_best_norm < 0.5
    
    # Simpler: use a smooth step function
    # When f_best_norm is close to 1 (early), explore more
    # When f_best_norm is close to 0 (late), exploit more
    
    # alpha_lcb = 0.5 * (1 + f_best_norm)  # ranges from 0.5 to 1.0
    # alpha_ei = 0.5 * (1 - f_best_norm)   # ranges from 0.5 to 0.0
    # But we want more dramatic transition
    
    # Use a power law for sharper transition
    # alpha_lcb = f_best_norm ** 0.5  # sqrt for sharper transition
    # alpha_ei = 1 - alpha_lcb
    
    # Actually, let's use a smoother approach:
    # alpha_ei = 1 - f_best_norm ** 1.5
    # alpha_lcb = f_best_norm ** 1.5
    # This gives:
    # f_best_norm = 1.0 -> alpha_ei = 0, alpha_lcb = 1.0 (pure exploration)
    # f_best_norm = 0.5 -> alpha_ei = 0.85, alpha_lcb = 0.15 (mostly exploitation)
    # f_best_norm = 0.0 -> alpha_ei = 1.0, alpha_lcb = 0.0 (pure exploitation)
    
    # Let's use a more balanced approach:
    # alpha_ei = 1 - 0.8 * f_best_norm
    # alpha_lcb = 0.8 * f_best_norm
    # This keeps at least 20% weight on each
    
    # Actually, the simplest and most robust is a 50/50 blend
    # which has been shown to work well in the inspiration
    
    # Let's use an adaptive blend that starts with more exploration
    # and transitions to more exploitation
    
    # Use a smooth sigmoid transition
    # t = 1 / (1 + exp(-5 * (f_best_norm - 0.3)))
    # This gives t ~ 0 when f_best_norm < 0.3, t ~ 1 when f_best_norm > 0.3
    
    # alpha_ei = 0.3 + 0.7 * (1 - t)
    # alpha_lcb = 0.7 * t
    
    # Simpler: just use the proven 50/50 blend from inspiration 1
    # which scored 1.4511, better than the current 1.3859
    
    alpha = 0.5
    score = alpha * ei_cost + (1 - alpha) * lcb_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
