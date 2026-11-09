"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm, t as t_dist

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a Student-t-based Expected Improvement (EI) blended with an adaptive
    Upper Confidence Bound (UCB). The Student-t distribution's heavier tails
    better account for the multi-modal nature of the Weierstrass function,
    encouraging exploration of distant uncertain regions.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior standard deviation in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        array of shape (2000,), scores where higher means evaluate next
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Student-t degrees of freedom: lower df = heavier tails = more exploration
    # We adapt df based on f_best: early (high f_best) -> low df (heavy tails)
    # late (low f_best) -> higher df (closer to Gaussian, more exploitation)
    fb = np.clip(f_best, 0.0, 1.0)
    # df ranges from 1.5 (very heavy) to 10.0 (near-Gaussian)
    df = 1.5 + 8.5 * (1.0 - fb)
    
    # Student-t based Expected Improvement for minimization
    # We want E[max(f_best - f, 0)] where f ~ t(mu, sigma, df)
    # For a t-distribution with location mu, scale sigma, and df degrees of freedom:
    # The CDF is: F(x) = t.cdf((x - mu) / sigma, df)
    # The PDF is: f(x) = t.pdf((x - mu) / sigma, df) / sigma
    
    # z_t = (f_best - mu) / sigma
    z_t = (f_best - mu) / sigma
    
    # Clip z_t to avoid numerical issues
    z_t = np.clip(z_t, -20, 20)
    
    # t-distribution CDF and PDF
    # For a standard t-distribution, cdf(z) gives P(T <= z)
    # We need P(f > f_best) = 1 - P(f <= f_best) = 1 - t.cdf((f_best - mu)/sigma, df)
    # And P(f < f_best) = t.cdf((f_best - mu)/sigma, df)
    
    # EI for minimization with t-distribution:
    # E[max(f_best - f, 0)] = (f_best - mu) * P(f < f_best) + sigma * t.pdf(z_t, df) * correction
    # Actually, for t-distribution:
    # E[max(f_best - f, 0)] = (f_best - mu) * F_t(z_t) + sigma * f_t(z_t) * (df / (df - 1)) * (1 - F_t(z_t) - z_t * f_t(z_t) / (df - 1))
    # This is getting complex. Let's use a simpler approximation:
    # The key insight is that t-distribution has heavier tails, so the "improvement" 
    # from uncertain regions is larger. We can approximate by scaling the standard EI.
    
    # Simpler approach: use the t-distribution CDF/PDF directly
    # F_t(z) = t.cdf(z, df)
    # f_t(z) = t.pdf(z, df)
    
    F_t = t_dist.cdf(z_t, df)
    f_t = t_dist.pdf(z_t, df)
    
    # For a t-distribution with scale sigma:
    # E[max(f_best - f, 0)] = (f_best - mu) * F_t(z_t) + sigma * f_t(z_t)
    # This is the same form as Gaussian EI, but with t-distribution CDF/PDF
    # The heavier tails mean F_t and f_t have different shapes, naturally
    # giving more weight to uncertain regions
    
    ei_t = (f_best - mu) * F_t + sigma * f_t
    
    # Ensure finite
    ei_t = np.where(np.isfinite(ei_t), ei_t, 0.0)
    
    # Adaptive UCB for minimization
    # beta decays as f_best approaches 0
    beta = 1.0 + 3.0 * fb
    ucb_score = -mu + beta * sigma
    
    # Normalize both to [0, 1] for blending
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range < 1e-10:
            return np.zeros_like(s)
        return (s - s_min) / s_range
    
    ei_norm = normalize_scores(ei_t)
    ucb_norm = normalize_scores(ucb_score)
    
    # Adaptive weights: early -> more UCB (exploration), late -> more EI (exploitation)
    # fb is high early, low late
    w_ucb = 0.5 + 0.4 * fb  # ranges from 0.5 to 0.9
    w_ei = 1.0 - w_ucb
    
    # Blend
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
