import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute normal PDF and CDF
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    from scipy.special import erf
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound for minimization: -mu + kappa*sigma
    # Higher kappa means more exploration
    kappa = 1.5
    lcb = -mu + kappa * sigma
    lcb = np.maximum(lcb, 0.0)
    
    # Upper Confidence Bound for minimization: -mu - kappa*sigma (penalizes high uncertainty)
    # Actually for minimization, we want to avoid high uncertainty when exploiting
    # Let's use a different approach: exploit score = -(mu - f_best) when sigma is small
    
    # Exploitation score: how much better than f_best could this be, weighted by certainty
    # Use negative expected value with uncertainty penalty
    exploitation = -(mu - f_best) - 0.5 * sigma
    exploitation = np.maximum(exploitation, 0.0)
    
    # Cost-aware normalization: divide by cost^beta
    # Beta should be moderate to balance cost without being too aggressive
    beta = 1.0
    cost_factor = cost_safe ** beta
    
    # Compute each acquisition divided by cost
    ei_score = ei / cost_factor
    lcb_score = lcb / cost_factor
    exploit_score = exploitation / cost_factor
    
    # Adaptive weighting based on convergence
    # If mu is close to f_best everywhere, we're converged -> exploit more
    # If mu is spread out, explore more
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min if mu_max > mu_min else 1e-9
    
    # How close is the best mu to f_best?
    best_mu_diff = f_best - np.min(mu)
    # Normalize by typical scale
    typical_scale = np.std(mu) if np.std(mu) > 1e-9 else 1.0
    convergence = np.clip(-best_mu_diff / (typical_scale + 1e-9), 0, 1)
    # convergence = 0 when exploring (best mu much worse than f_best impossible)
    # Actually if mu_min < f_best, we've found something better, but f_best is min observed
    # So mu_min should be >= f_best typically. Let's use sigma instead.
    
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    # High relative sigma means we're early/exploring
    if sigma_max > 1e-9:
        explore_ratio = sigma_mean / sigma_max
    else:
        explore_ratio = 0.0
    
    # Weights: more exploration early, more exploitation later
    # Use sigma to estimate stage: high sigma = early
    w_explore = 0.3 + 0.4 * explore_ratio  # weight for LCB (exploration)
    w_exploit = 0.3 + 0.4 * (1.0 - explore_ratio)  # weight for exploitation
    w_ei = 0.4  # fixed weight for EI
    
    # Normalize weights
    total_w = w_explore + w_exploit + w_ei
    w_explore /= total_w
    w_exploit /= total_w
    w_ei /= total_w
    
    # Weighted combination
    scores = w_ei * ei_score + w_explore * lcb_score + w_exploit * exploit_score
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
