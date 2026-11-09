"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a structural shift to Quantile of the Posterior (QoP) with adaptive
confidence, blended with Expected Improvement (EI) for exploitation,
normalized by cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erfc

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _fast_ei(mu, sigma, f_best):
    """
    Fast calculation of Expected Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    phi_z = 0.5 * erfc(-z / np.sqrt(2.0))
    
    # pdf(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    pdf_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei

def _quantile_of_posterior(mu, sigma, p):
    """
    Calculate the p-th quantile of the posterior distribution.
    For minimization, we want to maximize the likelihood of finding a low value.
    The p-th quantile is mu + sigma * z_p.
    Since we want to find low values, we look at the lower tail (p < 0.5).
    A lower quantile value is better.
    So we return -q_p to maximize it.
    """
    # Precompute z_p for standard normal
    # p=0.10 -> z ~ -1.28
    # p=0.25 -> z ~ -0.67
    # p=0.50 -> z = 0.0
    # We will compute this dynamically or use a lookup/approximation.
    # For speed, we can use scipy.stats.norm.ppf but that might be slow in a loop?
    # No, it's a scalar operation here effectively if p is fixed, but we might want to adapt p.
    # Let's just use a hardcoded value for the specific p we choose, or compute once.
    # The strategy says "e.g., 90th percentile for minimization" which is confusing phrasing.
    # Usually QoP for minimization uses the lower quantile (e.g., 10th percentile).
    # Let's use p=0.25 (25th percentile) as a balance.
    # z_0.25 = -0.67448975
    
    # Actually, let's allow adaptive p.
    # If f_best is high, we are uncertain, maybe look further into the tail?
    # If f_best is low, we are confident, look closer to the mean?
    
    # To avoid calling scipy.stats.norm.ppf repeatedly (which is fine but let's be safe),
    # we can approximate or just use a fixed value if we don't adapt p.
    # Let's adapt p based on f_best.
    # Early: p=0.1 (more exploration of tails)
    # Late: p=0.4 (closer to mean, exploitation)
    
    # z-values for common percentiles:
    # 0.05: -1.645
    # 0.10: -1.282
    # 0.25: -0.674
    # 0.50: 0.0
    
    # We'll compute z_p using a simple approximation or just hardcode the logic.
    # Since p is a scalar, we can call norm.ppf once per acquisition call.
    from scipy.stats import norm
    z_p = norm.ppf(p)
    
    q_p = mu + sigma * z_p
    
    # We want to minimize q_p, so we maximize -q_p
    return -q_p

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Structural change: Primary driver is Quantile of the Posterior (QoP),
    blended with EI for fine-grained exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Fallback: simple cost-aware QoP with fixed p=0.25
        cost_safe = np.maximum(cost, 1e-9)
        z_025 = -0.6745
        q_025 = mu + sigma * z_025
        scores = -q_025 / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Determine adaptive quantile probability p
    # Rastrigin has many local minima.
    # High f_best (>10): We are likely far from the global min. Explore the tails.
    # Low f_best (<1): We are close. Exploit the mean.
    if f_best > 10.0:
        p = 0.10  # 10th percentile (z ~ -1.28)
    elif f_best > 5.0:
        p = 0.20  # 20th percentile (z ~ -0.84)
    elif f_best > 2.0:
        p = 0.30  # 30th percentile (z ~ -0.52)
    else:
        p = 0.40  # 40th percentile (z ~ -0.25)
        
    # 2. Calculate QoP score
    # -q_p = -(mu + sigma * z_p) = -mu - sigma * z_p
    # Since z_p is negative, -sigma * z_p is positive.
    # So high sigma increases the score (exploration).
    # Low mu increases the score (exploitation).
    from scipy.stats import norm
    z_p = norm.ppf(p)
    qop_score = -(mu + sigma * z_p)
    
    # 3. Calculate EI score
    ei_score = _fast_ei(mu, sigma, f_best)
    
    # 4. Normalize scores to comparable scales
    # QoP and EI have different magnitudes.
    # EI is bounded by f_best - min(mu) roughly.
    # QoP is unbounded below (if mu is very negative).
    # Let's normalize by the max of each to blend them fairly.
    
    qop_max = np.max(qop_score)
    ei_max = np.max(ei_score)
    
    # Avoid division by zero
    if qop_max > 1e-12:
        qop_norm = qop_score / qop_max
    else:
        qop_norm = np.zeros_like(qop_score)
        
    if ei_max > 1e-12:
        ei_norm = ei_score / ei_max
    else:
        ei_norm = np.zeros_like(ei_score)
        
    # 5. Blend QoP and EI
    # Early: QoP dominates (exploration of multimodal landscape)
    # Late: EI dominates (exploitation of best region)
    if f_best > 5.0:
        w_qop = 0.8
        w_ei = 0.2
    elif f_best > 2.0:
        w_qop = 0.6
        w_ei = 0.4
    else:
        w_qop = 0.4
        w_ei = 0.6
        
    combined = w_qop * qop_norm + w_ei * ei_norm
    
    # 6. Cost-aware normalization
    # Divide by cost to prefer cheaper evaluations.
    # Cost is 1 + mean(x_scaled). Range [1, 2].
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
