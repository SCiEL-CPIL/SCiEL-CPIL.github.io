"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_T0 = [None]
_CALLS = [0]


def _fast_norm_cdf(z):
    """Fast, accurate approximation of the standard normal CDF."""
    # Use Abramowitz & Stegun 7.1.26 for |z| < 5, then asymptotic
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    mask = np.abs(z) < 5.0
    # For small |z|, use the polynomial approximation
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z[mask]))
    poly = 1.0 - 0.3989422804014327 * t * (
        1.0 + t * (1.25331413 + t * (1.73410772 + t * (2.43753815 + t * 1.92856389)))
    )
    # Wait, that's not right. Let me use the standard A&S formula properly.
    # Phi(x) = 1 - phi(x) * (b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5) for x >= 0
    # where t = 1/(1+0.2316419*x)
    # This is for the upper tail. Let me just use a simpler approach.
    
    # Actually, let me use the erf-based approach which is fast in numpy
    # But scipy.special.erf might be faster than scipy.stats.norm.cdf
    # For now, let's just use a good polynomial for the full range
    
    # Simpler: use the fact that for our purposes, we need accuracy to ~1e-6
    # The A&S 7.1.26 formula:
    # For x >= 0: Phi(x) = 1 - (1/sqrt(2*pi)) * exp(-x^2/2) * (p1*t + p2*t^2 + p3*t^3 + p4*t^4 + p5*t^5)
    # where t = 1/(1 + 0.2316419*x)
    # p1=0.319381530, p2=-0.356563782, p3=1.781477937, p4=-1.821255978, p5=1.330274429
    
    x = z[mask]
    pos = x >= 0
    neg = ~pos
    
    if np.any(pos):
        xp = x[pos]
        t = 1.0 / (1.0 + 0.2316419 * xp)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        cdf_pos = 1.0 - np.exp(-0.5 * xp * xp) / np.sqrt(2.0 * np.pi) * poly
        result[mask][pos] = cdf_pos
    
    if np.any(neg):
        xn = x[neg]
        # Phi(-x) = 1 - Phi(x)
        t = 1.0 / (1.0 + 0.2316419 * (-xn))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        cdf_pos_neg = 1.0 - np.exp(-0.5 * xn * xn) / np.sqrt(2.0 * np.pi) * poly
        result[mask][neg] = 1.0 - cdf_pos_neg
    
    # For large |z|, use asymptotic
    if np.any(~mask):
        result[~mask & (z > 0)] = 1.0
        result[~mask & (z < 0)] = 0.0
    
    return result


def _fast_norm_pdf(z):
    """Fast standard normal PDF."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI with a time-aware exploration bonus.
    
    Args:
        mu: GP means, shape (2000,)
        sigma: GP std devs, shape (2000,)
        f_best: best observed value (scalar)
        cost: evaluation costs, shape (2000,)
        
    Returns:
        scores, shape (2000,), higher = evaluate next
    """
    # Deadline guard
    if _T0[0] is None:
        _T0[0] = time.time()
    _CALLS[0] += 1
    elapsed = time.time() - _T0[0]
    
    # Safety margin: 330s limit, stop heavy compute at 315s
    if elapsed > 315.0:
        s = np.maximum(sigma, 1e-9)
        c = np.maximum(cost, 1e-9)
        score = s / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Noise-aware lower bound: prevent over-confident exploitation
    # The GP noise is 1e-6, but we want a practical floor
    s = np.maximum(sigma, 1e-6)
    c = np.maximum(cost, 1e-9)
    
    # Expected Improvement (minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    Phi = _fast_norm_cdf(z)
    phi = _fast_norm_pdf(z)
    
    ei = (f_best - mu) * Phi + s * phi
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / c
    
    # Adaptive exploration: use UCB-like term with uncertainty
    # Gentler decay: maintain at least 10% exploration weight
    t_frac = min(elapsed / 280.0, 1.0)
    expl_weight = 0.25 * (0.1 + 0.9 * (1.0 - t_frac))
    
    # UCB term: sigma / cost
    ucb = s / c
    
    # Combine: cost-EI + exploration
    score = ei_cost + expl_weight * ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
