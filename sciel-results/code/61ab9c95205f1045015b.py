import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is non-negative and numerically stable
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard: safety margin before 300s limit
    deadline = start_time + 290.0
    
    # Knowledge Gradient (KG) for minimization:
    # KG(x) = E[max(f_best - min(f(x), f_best), 0)]
    #       = E[max(f_best - f(x), 0)] when f(x) < f_best
    # For a Gaussian posterior N(mu, sigma^2):
    # KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # This is mathematically identical to Expected Improvement (EI) for minimization,
    # but the framing emphasizes global optimality: we're computing the expected
    # reduction in the best-known value, not just local improvement.
    
    z = (f_best - mu) / sigma
    kg = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Thompson Sampling component:
    # Draw from posterior N(mu, sigma^2) and score by negative sample.
    # This provides natural exploration: high-sigma regions have wide distributions,
    # so they're more likely to produce very low samples.
    # Use a deterministic seed derived from the optimization state for reproducibility
    # within a call but varying across calls.
    seed_raw = abs(f_best) * 1e6 + np.sum(np.abs(mu)) + np.sum(sigma)
    seed = int(seed_raw) % (2**32)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mu.shape[0])
    samples = mu + sigma * eps
    ts_scores = -samples
    
    # Check deadline before normalization
    if time.time() > deadline:
        scores = np.where(np.isfinite(kg), kg, 0.0)
        return scores
    
    # Normalize both scores to zero mean, unit variance to balance scales
    kg_mean = np.mean(kg)
    kg_std = np.std(kg) + 1e-8
    kg_norm = (kg - kg_mean) / kg_std
    
    ts_mean = np.mean(ts_scores)
    ts_std = np.std(ts_scores) + 1e-8
    ts_norm = (ts_scores - ts_mean) / ts_std
    
    # Combine: Knowledge Gradient as the primary structural signal (global optimality),
    # Thompson Sampling as a refinement that adds stochastic exploration diversity.
    # The KG term directs search toward regions that maximize expected reduction in
    # the global best, while TS adds robustness against unlucky deterministic choices.
    alpha = 0.7  # Weight for Knowledge Gradient (exploitation of global structure)
    beta = 1.0 - alpha  # Weight for Thompson Sampling (stochastic exploration)
    
    scores = alpha * kg_norm + beta * ts_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
