import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle zero sigma and cost to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Thompson Sampling Component
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # For minimization, we want low y.
    # We use a single random draw for all points to create a consistent "sampled function"
    z_ts = np.random.standard_normal(n)
    sampled_f = mu + sigma_safe * z_ts
    
    # Score: We want to maximize the likelihood of finding a low value relative to cost.
    # Lower sampled_f is better. So we negate it.
    # Dividing by cost makes cheaper evaluations more attractive.
    ts_score_raw = -sampled_f / cost_safe
    
    # 2. Expected Improvement Component
    # EI = E[max(0, f_best - f(x))]
    # For minimization, improvement is f_best - f(x).
    imp = f_best - mu
    z_ei = imp / sigma_safe
    cdf_z = norm.cdf(z_ei)
    pdf_z = norm.pdf(z_ei)
    
    ei_raw = imp * cdf_z + sigma_safe * pdf_z
    ei_raw = np.maximum(ei_raw, 0.0)
    
    # Score: Higher EI is better. Penalize by cost.
    ei_score_raw = ei_raw / cost_safe
    
    # 3. Normalization
    # Normalize scores to [0, 1] to allow weighted sum
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range > 1e-12:
            return (s - s_min) / s_range
        return np.zeros_like(s)
    
    ts_norm = normalize_scores(ts_score_raw)
    ei_norm = normalize_scores(ei_score_raw)
    
    # 4. Adaptive Weighting
    # We want to trust Thompson Sampling more when uncertainty is high (early/multimodal regions)
    # and trust EI more when we are close to the optimum (exploitation).
    # Measure uncertainty relative to the best found so far.
    # If sigma is large compared to the spread of mu, we are in an exploratory phase.
    med_sigma = np.median(sigma_safe)
    spread_mu = np.max(mu) - np.min(mu)
    
    # If spread is 0, we are likely in a very tight region or early on with identical means (unlikely with GP).
    if spread_mu > 1e-9:
        uncertainty_ratio = med_sigma / spread_mu
    else:
        uncertainty_ratio = 1.0
        
    # Map uncertainty_ratio to a weight for TS.
    # High uncertainty -> High TS weight (0.8)
    # Low uncertainty -> Lower TS weight (0.4), favoring EI
    w_ts = 0.4 + 0.4 * np.clip(uncertainty_ratio, 0.0, 1.0)
    w_ei = 1.0 - w_ts
    
    # 5. Combine Scores
    scores = w_ts * ts_norm + w_ei * ei_norm
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
