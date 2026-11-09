import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample a function from the posterior
    # For minimization, we sample f(x) = mu(x) - sigma(x) * z where z ~ N(0,1)
    # This gives a stochastic sample of the objective value
    # We want to find the candidate that minimizes f(x) / cost(x)
    # To convert to "higher is better" acquisition: -f(x) / cost(x)
    
    # Generate random z values for Thompson Sampling
    rng = np.random.default_rng()
    z_sample = rng.standard_normal(2000)
    
    # Sampled function values (lower is better for minimization)
    sampled_f = mu - sigma * z_sample
    
    # Cost-aware Thompson Sampling score: -sampled_f / cost
    # Higher score means we prefer this candidate
    ts_score = -sampled_f / cost_safe
    
    # Also compute EI for a blend
    z_ei = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z_ei)
    pdf_z = norm.pdf(z_ei)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    ei_per_cost = ei / cost_safe
    
    # Normalize both scores
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range > 1e-12:
            return (s - s_min) / s_range
        return np.zeros_like(s)
    
    norm_ts = normalize_scores(ts_score)
    norm_ei = normalize_scores(ei_per_cost)
    
    # Blend Thompson Sampling with EI
    # Thompson Sampling naturally balances exploration and exploitation
    # Use a moderate blend to keep some exploitation bias
    w_ts = 0.7
    w_ei = 0.3
    
    scores = w_ts * norm_ts + w_ei * norm_ei
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
    
    return scores
