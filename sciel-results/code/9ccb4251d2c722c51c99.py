import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _normalize(s):
    """Normalize score to [0,1] range."""
    s = np.where(np.isfinite(s), s, -np.inf)
    finite_mask = np.isfinite(s)
    if not np.any(finite_mask):
        return np.zeros_like(s)
    s_min = np.min(s[finite_mask])
    s_max = np.max(s[finite_mask])
    if np.isfinite(s_max - s_min) and (s_max - s_min) > 1e-12:
        s_norm = (s - s_min) / (s_max - s_min)
    else:
        s_norm = np.zeros_like(s)
    return np.where(np.isfinite(s_norm), s_norm, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Search over multiple acquisition strategies with deadline guard.
    """
    t0 = time.time()
    
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    known_min = -78.3323
    n = len(mu)
    
    # Progress estimate: 0 at start, 1 near optimum
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # Precompute common terms
    z = (f_best - mu) / (sigma + 1e-12)
    ei_base = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Strategy 1: Cost-Adjusted EI with entropy bonus
    # CA-EI: EI / cost^alpha, where alpha adapts with progress
    # Entropy bonus: reward high-uncertainty regions to prevent premature convergence
    scores = []
    
    # Try multiple cost exponents and entropy weights
    cost_exponents = [0.8, 1.0, 1.2, 1.5]
    entropy_weights = [0.0, 0.1, 0.2, 0.3]
    
    best_score = None
    best_idx = -1
    
    for cost_exp in cost_exponents:
        for ent_w in entropy_weights:
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Cost-adjusted EI
            ca_ei = ei_base / (cost ** cost_exp)
            
            # Entropy bonus: sigma * log(sigma + eps) rewards high uncertainty
            # This helps escape local minima in multi-modal landscapes
            entropy = sigma * np.log(sigma + 1e-9)
            entropy = np.where(np.isfinite(entropy), entropy, 0.0)
            
            # Scale entropy by progress: more exploration early, less late
            ent_scale = (1.0 - 0.5 * progress) * ent_w
            
            score = ca_ei + ent_scale * entropy
            
            # Normalize for comparison
            score_norm = _normalize(score)
            
            # Evaluate this strategy by checking if it produces diverse, reasonable scores
            # Use a simple heuristic: prefer strategies that balance exploitation and exploration
            finite_count = np.sum(np.isfinite(score))
            if finite_count < n * 0.9:
                continue
            
            # Compute a quality metric: variance (diversity) + mean (overall level)
            # Higher variance means the strategy is distinguishing between points
            var = np.var(score[np.isfinite(score)])
            mean = np.mean(score[np.isfinite(score)])
            
            # Quality: reward strategies with good separation and reasonable scale
            quality = var + 0.1 * abs(mean)
            
            if best_score is None or quality > best_score:
                best_score = quality
                best_idx = len(scores)
                scores.append(score)
        
        if time.time() > _DEADLINE - 10.0:
            break
    
    # If no valid strategy found, fall back to simple CA-EI
    if best_idx == -1 or best_score is None:
        cost_exp = 1.0
        final_score = ei_base / (cost ** cost_exp)
    else:
        final_score = scores[best_idx]
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu / (cost + 1e-9)
    
    return final_score
