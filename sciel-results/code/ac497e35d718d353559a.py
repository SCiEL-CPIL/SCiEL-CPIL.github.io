"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization with deadline guard.
    # Score(x) = -mu(x) + kappa * sigma(x)
    # Higher score means evaluate next.
    
    # Deadline guard: ensure we return well before 330s wall-clock limit
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive kappa: reduce exploration as we find better solutions
    # and when uncertainty is low relative to the function range
    mu_range = np.ptp(mu) if len(mu) > 0 else 1.0
    sigma_mean = np.mean(sigma) if len(sigma) > 0 else 1.0
    
    # Normalize f_best relative to typical Ackley values (range ~0-10)
    # As f_best approaches 0, we should exploit more
    f_best_norm = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # When f_best is close to optimum, reduce exploration
    exploit_factor = 1.0 - 0.5 * (1.0 - f_best_norm)
    
    # Base kappa between 0.5 and 1.5, scaled by exploit factor
    kappa = (0.5 + 1.0 * f_best_norm) * exploit_factor
    
    # If uncertainty is very low overall, increase exploration slightly
    if sigma_mean < 1e-3 and mu_range > 1e-3:
        kappa *= 1.2
    
    # Compute base LCB score
    scores = -mu + kappa * sigma
    
    # Check deadline after base computation
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Hypervolume-inspired diversity refinement:
    # Score candidates by how much they extend the Pareto front of (mu, sigma)
    # This encourages diverse trade-offs rather than pure exploitation/exploration
    
    # Normalize mu and sigma to [0, 1] for Pareto front computation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # For minimization, lower mu and lower sigma are better
    # Pareto front: points not dominated by others (lower or equal in both, strictly lower in at least one)
    # We compute a diversity score based on hypervolume contribution
    
    # Use a subsample for efficiency if needed
    n = len(mu)
    if n > 500:
        # Use random subsample for Pareto front computation
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=500, replace=False)
        mu_sub = mu_norm[idx]
        sigma_sub = sigma_norm[idx]
    else:
        idx = np.arange(n)
        mu_sub = mu_norm
        sigma_sub = sigma_norm
    
    # Compute Pareto front of (mu, sigma) - lower is better for both
    # A point is non-dominated if no other point has both mu <= and sigma <= with at least one strict
    pareto_mask = np.ones(len(mu_sub), dtype=bool)
    for i in range(len(mu_sub)):
        if not pareto_mask[i]:
            continue
        # Check if any other point dominates this one
        dominated = (mu_sub <= mu_sub[i]) & (sigma_sub <= sigma_sub[i])
        dominated[i] = False  # Don't compare with self
        if np.any(dominated & pareto_mask):
            pareto_mask[i] = False
    
    # Compute hypervolume contribution for each point
    # For each point, compute the area of the rectangle it uniquely covers
    # This is an approximation - we use the distance to the nearest non-dominated point
    hv_scores = np.zeros(n)
    
    if np.sum(pareto_mask) > 0:
        # For each point, compute its contribution to the hypervolume
        # Simple approximation: distance to nearest non-dominated point in normalized space
        for i in idx:
            if pareto_mask[list(idx).index(i)]:
                # This is on the Pareto front, give it a boost
                hv_scores[i] = 0.5
            else:
                # Compute distance to nearest Pareto point
                pareto_idx = idx[pareto_mask]
                if len(pareto_idx) > 0:
                    dists = np.sqrt((mu_norm[i] - mu_norm[pareto_idx])**2 + 
                                    (sigma_norm[i] - sigma_norm[pareto_idx])**2)
                    min_dist = np.min(dists)
                    # Points closer to Pareto front get higher diversity score
                    hv_scores[i] = max(0, 0.5 * (1 - min_dist))
    
    # Blend the LCB score with the hypervolume diversity score
    # Weight the diversity component more in early stages when f_best is high
    diversity_weight = 0.2 + 0.3 * f_best_norm
    
    # Normalize hv_scores to [0, 1]
    hv_max = np.max(hv_scores) if np.max(hv_scores) > 0 else 1.0
    hv_norm = hv_scores / hv_max
    
    # Final score: blend of LCB and diversity
    # Scale LCB scores to be comparable
    lcb_min, lcb_max = np.min(scores), np.max(scores)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = (scores - lcb_min) / lcb_range
    
    scores = (1 - diversity_weight) * lcb_norm + diversity_weight * hv_norm
    
    # Check deadline
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Refine: dampen extreme uncertainty to focus on promising uncertain regions
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
        # Dampen scores where normalized sigma is very high (>0.95)
        # This keeps exploration but avoids pure noise regions
        damp_mask = sigma_norm > 0.95
        if np.any(damp_mask):
            # Apply a gentle penalty proportional to how far above threshold
            excess = (sigma_norm[damp_mask] - 0.95) / 0.05
            scores[damp_mask] -= 0.1 * excess
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
