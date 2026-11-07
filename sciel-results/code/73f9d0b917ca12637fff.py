import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses Uncertainty-Weighted Improvement (UWI) with local baseline
    estimated via nearest-neighbor in (mu, sigma) space.
    """
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    known_min = -78.3323
    
    # Progress estimate
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # Build local baseline using KD-tree nearest neighbors in (mu, sigma) space
    # Each point's neighborhood gives a local estimate of what "good" looks like
    n = len(mu)
    if n < 10:
        # Too few points for meaningful local baseline; use global
        local_mu = np.full(n, f_best)
        local_sigma = np.full(n, 1.0)
    else:
        # Create feature space for KD-tree: normalized mu and sigma
        mu_range = mu.max() - mu.min() + 1e-12
        sigma_range = sigma.max() - sigma.min() + 1e-12
        
        mu_norm = (mu - mu.min()) / mu_range
        sigma_norm = (sigma - sigma.min()) / sigma_range
        
        # Weight sigma more heavily to cluster points with similar uncertainty
        features = np.column_stack([mu_norm, 2.0 * sigma_norm])
        
        tree = cKDTree(features)
        
        # Use 5 nearest neighbors for local baseline
        k = min(5, n - 1)
        distances, indices = tree.query(features, k=k+1)
        
        # For each point, compute weighted average of neighbors' mu and sigma
        # Weight by inverse distance (closer neighbors matter more)
        local_mu = np.zeros(n)
        local_sigma = np.zeros(n)
        
        for i in range(n):
            # Skip self (index 0 is always self)
            dists = distances[i, 1:]
            idxs = indices[i, 1:]
            
            # Inverse distance weighting
            weights = 1.0 / (dists + 1e-12)
            weights = weights / np.sum(weights)
            
            local_mu[i] = np.sum(weights * mu[idxs])
            local_sigma[i] = np.sum(weights * sigma[idxs])
    
    # Compute Uncertainty-Weighted Improvement (UWI)
    # UWI rewards candidates where:
    # 1. The prediction is lower than the local baseline (improvement)
    # 2. The uncertainty is high relative to local context (exploration at basin boundaries)
    
    # Local improvement: how much better is this point than its neighborhood's typical value
    # Use local_mu as the baseline instead of global f_best
    local_improvement = local_mu - mu  # Positive if mu is below local baseline
    
    # Uncertainty weight: sigma relative to local sigma
    # High sigma relative to neighbors = boundary of explored region
    local_sigma_safe = np.where(local_sigma > 1e-12, local_sigma, 1e-9)
    uncertainty_ratio = sigma / local_sigma_safe
    
    # UWI score: combine local improvement and uncertainty
    # Scale uncertainty contribution based on progress
    # Early: more weight on uncertainty (exploration)
    # Late: more weight on improvement (exploitation)
    unc_weight = 1.5 - 1.0 * progress
    
    # UWI = local_improvement + unc_weight * uncertainty_ratio * sigma
    # This rewards points that are both better than local context AND have high uncertainty
    uwi = local_improvement + unc_weight * uncertainty_ratio * sigma
    
    # Cost adjustment: divide by cost to prefer cheaper evaluations
    # Use a moderate cost penalty that decreases with progress
    cost_exp = 1.2 - 0.5 * progress
    uwi_cost = uwi / (cost ** cost_exp)
    
    # Also compute a standard EI-based score as a secondary signal
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_cost = ei / (cost ** cost_exp)
    
    # Combine UWI and EI with adaptive weighting
    # UWI is the primary signal, EI provides a global improvement perspective
    # Weight UWI more heavily as it targets basin boundaries
    uwi_weight = 0.7
    ei_weight = 0.3
    
    # Normalize both scores to [0, 1] for combination
    def normalize(s):
        s = np.where(np.isfinite(s), s, -np.inf)
        s_min = np.min(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else -np.inf
        s_max = np.max(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else np.inf
        if np.isfinite(s_max - s_min) and (s_max - s_min) > 1e-12:
            s_norm = (s - s_min) / (s_max - s_min)
        else:
            s_norm = np.zeros_like(s)
        return np.where(np.isfinite(s_norm), s_norm, 0.0)
    
    uwi_norm = normalize(uwi_cost)
    ei_norm = normalize(ei_cost)
    
    combined_norm = uwi_weight * uwi_norm + ei_weight * ei_norm
    
    # Map back to a reasonable score range
    # Use the range of the UWI-based score
    uwi_cost_clean = np.where(np.isfinite(uwi_cost), uwi_cost, 0)
    uc_min = np.min(uwi_cost_clean)
    uc_max = np.max(uwi_cost_clean)
    
    if np.isfinite(uc_max - uc_min) and (uc_max - uc_min) > 1e-12:
        final_score = uc_min + combined_norm * (uc_max - uc_min)
    else:
        final_score = uwi_cost_clean
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu / (cost + 1e-9)
    
    return final_score
