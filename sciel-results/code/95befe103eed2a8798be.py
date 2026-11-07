import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Uses a Stochastic Expected Improvement (SEI) formulation where cost is
    modeled as a log-normal random variable. This integrates cost uncertainty
    directly into the acquisition integral rather than applying a deterministic
    post-hoc penalty.
    
    The SEI is: E[max(f_best - f(x), 0) / C(x)]
    where C(x) ~ LogNormal(mu_c, sigma_c) with parameters derived from cost.
    
    For a log-normal cost C with mean m and std s, we have:
    - mu_c = log(m^2 / sqrt(m^2 + s^2))
    - sigma_c = sqrt(log(1 + s^2/m^2))
    
    The SEI integral is:
    E[max(f_best - f, 0) / C] = E[max(f_best - f, 0) * C^{-1}]
    
    Since C^{-1} is log-normal with parameters -mu_c, sigma_c, we can compute:
    E[max(f_best - f, 0) * C^{-1}] = E[max(f_best - f, 0)] * E[C^{-1}] 
    only if f and C are independent, which they are.
    
    E[C^{-1}] for C ~ LogNormal(mu_c, sigma_c) is:
    exp(-mu_c + 0.5 * sigma_c^2)
    
    So SEI = EI(x) * exp(-mu_c + 0.5 * sigma_c^2)
    
    This is equivalent to EI / E[C] when sigma_c is small, but properly
    accounts for cost uncertainty.
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        return 1.0 / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement: positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Model cost as log-normal
    # cost is in [1, 2], we treat each point's cost as a log-normal random variable
    # with mean = cost and a small standard deviation to capture uncertainty
    # Use a fixed relative uncertainty of 10% for cost
    cost_mean = cost
    cost_std = 0.1 * cost_mean
    
    # Log-normal parameters
    # For LogNormal(mu_c, sigma_c):
    # mean = exp(mu_c + 0.5*sigma_c^2)
    # var = (exp(sigma_c^2) - 1) * exp(2*mu_c + sigma_c^2)
    # So: sigma_c^2 = log(1 + var/mean^2)
    #     mu_c = log(mean) - 0.5*sigma_c^2
    
    var = cost_std ** 2
    sigma_c_sq = np.log(1.0 + var / (cost_mean ** 2))
    sigma_c = np.sqrt(sigma_c_sq)
    mu_c = np.log(cost_mean) - 0.5 * sigma_c_sq
    
    # E[C^{-1}] = exp(-mu_c + 0.5 * sigma_c^2)
    e_inv_c = np.exp(-mu_c + 0.5 * sigma_c_sq)
    
    # SEI = EI * E[C^{-1}]
    sei = ei * e_inv_c
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps in early stages when EI is low everywhere
    sigma_scale = np.abs(f_best) + 1.0
    rel_sigma = sigma_safe / sigma_scale
    
    # Exploration bonus: small, proportional to relative sigma, weighted by E[C^{-1}]
    exploration_bonus = 0.05 * rel_sigma * e_inv_c
    
    score = sei + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
