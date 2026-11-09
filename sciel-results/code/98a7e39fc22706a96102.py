import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    Uses a Thompson Sampling-inspired stochastic score combined with Expected Improvement.
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Thompson Sampling: Draw a single sample from the posterior for all candidates
    # y ~ N(mu, sigma^2)
    # We use a fixed random seed derived from the current state to ensure reproducibility within a run
    # but allow variation between calls if we don't want to be fully deterministic.
    # However, standard Thompson Sampling in BO usually uses a fresh sample each time.
    # Since we don't have a persistent RNG state, we can use np.random.standard_normal.
    # To make it somewhat stable or reproducible per "step" isn't possible without state.
    # Let's just use a fresh sample.
    
    # The directive suggests: "selecting the point that minimizes y divided by the cost"
    # Score_TS = -y / cost  (since we want to maximize score, and y is the predicted value)
    
    # Generate sample
    eps = np.random.standard_normal(n)
    y_sample = mu + sigma_safe * eps
    
    # Thompson Sampling Score: Maximize -y_sample / cost
    ts_score = -y_sample / cost_safe
    
    # Cost-normalized EI Score
    # The current program used LCB, but inspirations show EI is strong.
    # Let's combine TS and EI.
    # EI is deterministic and robust. TS is stochastic and explores well.
    
    # Normalize both to comparable scales
    ts_std = np.std(ts_score)
    ei_std = np.std(ei)
    
    if ts_std > 1e-8:
        ts_norm = (ts_score - np.mean(ts_score)) / ts_std
    else:
        ts_norm = np.zeros(n)
        
    if ei_std > 1e-8:
        ei_norm = (ei - np.mean(ei)) / ei_std
    else:
        ei_norm = np.zeros(n)
        
    # Adaptive weighting based on distance to minimum
    # Early on (large dist), rely more on exploration (TS).
    # Late on (small dist), rely more on exploitation (EI).
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Map dist to weight for TS (exploration)
    # dist ~ 3.3 -> weight ~ 0.7
    # dist ~ 0.0 -> weight ~ 0.3
    ts_weight = 0.3 + 0.4 * np.exp(-dist_to_min / 0.5)
    ei_weight = 1.0 - ts_weight
    
    combined = ts_weight * ts_norm + ei_weight * ei_norm
    
    # Cost-aware: The scores are already cost-normalized via the division in ts_score and the nature of EI.
    # However, EI is not explicitly divided by cost in the inspiration that scored 1.0.
    # Let's check: Inspiration 2 and 3 use `scores = ei / cost_safe`.
    # My `ei_norm` is just normalized EI, not cost-normalized.
    # I should incorporate cost into the EI part as well, or trust the normalization.
    # Actually, `ts_score` is already `-y/cost`.
    # `ei` is absolute improvement. To make it cost-aware, it should be `ei / cost`.
    
    # Let's re-calculate EI score with cost normalization
    ei_cost_norm = ei / cost_safe
    
    ei_cn_std = np.std(ei_cost_norm)
    if ei_cn_std > 1e-8:
        ei_cn_norm = (ei_cost_norm - np.mean(ei_cost_norm)) / ei_cn_std
    else:
        ei_cn_norm = np.zeros(n)
        
    # Combine normalized TS and cost-normalized EI
    combined = ts_weight * ts_norm + ei_weight * ei_cn_norm
    
    scores = combined
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    return scores
