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
    
    # Calculate z-score for Expected Improvement (Minimization)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    poi = cdf_z
    
    # Multi-strategy ensemble:
    # 1. Cost-aware EI
    ei_per_cost = ei / cost_safe
    
    # 2. Cost-aware UCB (exploration)
    # UCB = mu - beta * sigma (for minimization)
    # We want to maximize (mu - beta*sigma) / cost
    # But we need to be careful: lower mu is better, higher sigma is more uncertain
    # UCB for minimization: mu - beta * sigma (lower is better)
    # To convert to "higher is better" for acquisition: -mu + beta*sigma
    # But we're working with candidates, so we want to find points where -mu + beta*sigma is high
    # Actually, let's think differently: we want to explore where sigma is high
    # UCB score = (beta * sigma - mu) / cost, but this could be dominated by -mu
    # Let's use a normalized approach
    
    # Strategy 1: EI/cost
    score_ei = ei_per_cost
    
    # Strategy 2: Pure exploitation - best predicted value per cost
    # For minimization, lower mu is better. So -mu/cost gives higher score to lower mu
    score_exploit = -mu / cost_safe
    
    # Strategy 3: Exploration - uncertainty per cost
    score_explore = sigma / cost_safe
    
    # Strategy 4: PoI per cost
    score_poi = poi / cost_safe
    
    # Normalize each score to [0, 1] range for fair combination
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range > 1e-12:
            return (s - s_min) / s_range
        return np.zeros_like(s)
    
    norm_ei = normalize_scores(score_ei)
    norm_exploit = normalize_scores(score_exploit)
    norm_explore = normalize_scores(score_explore)
    norm_poi = normalize_scores(score_poi)
    
    # Adaptive weights based on progress
    # Early on: more exploration, later: more exploitation
    # Use the spread of mu as a proxy for how much we've learned
    mu_spread = np.max(mu) - np.min(mu)
    sigma_mean = np.mean(sigma)
    
    # When sigma_mean is small relative to mu_spread, we're more confident -> exploit more
    if mu_spread > 1e-12:
        confidence = min(1.0, sigma_mean / (mu_spread + 1e-12))
    else:
        confidence = 0.5
    
    # Weight allocation:
    # EI: always important
    # Exploit: increase with confidence
    # Explore: decrease with confidence
    # PoI: moderate
    w_ei = 0.4
    w_exploit = 0.25 * (1.0 - confidence) + 0.35 * confidence
    w_explore = 0.25 * (1.0 - confidence) + 0.15 * (1.0 - confidence)
    w_poi = 0.1
    
    # Normalize weights
    w_sum = w_ei + w_exploit + w_explore + w_poi
    w_ei /= w_sum
    w_exploit /= w_sum
    w_explore /= w_sum
    w_poi /= w_sum
    
    # Combine
    scores = w_ei * norm_ei + w_exploit * norm_exploit + w_explore * norm_explore + w_poi * norm_poi
    
    # Add a small raw EI component to preserve scale information
    scores += 0.05 * norm_ei
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
    
    return scores
