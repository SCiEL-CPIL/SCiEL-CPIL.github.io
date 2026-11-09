"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Set-based greedy selection maximizing Joint Expected Improvement (J-EI).
    
    Strategy:
    1. Compute pointwise Expected Improvement (EI).
    2. Estimate spatial correlations between candidates using a Gaussian kernel
       on normalized mu and sigma values (proxy for input space proximity).
    3. Greedily select a diverse batch of K candidates that maximizes the 
       sum of J-EI contributions, penalized by cost.
    4. Assign high scores to selected candidates and low scores to others,
       creating a "set-based" priority that encourages the evaluator to pick
       diverse, high-value points rather than redundant local optima.
    """
    if _check_deadline():
        # Fallback: standard EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Base score: EI / cost (cost-awareness)
    base_score = ei / cost_safe
    
    # 3. Estimate Spatial Correlation (Proxy for J-EI)
    # We model the correlation between candidates i and j as:
    # rho_ij = exp(-d_ij^2 / (2 * l^2))
    # where d_ij is a distance in the (mu, sigma) feature space.
    
    # Normalize features to [0, 1]
    mu_min, mu_max = np.min(mu), np.max(mu)
    sig_min, sig_max = np.min(sigma_safe), np.max(sigma_safe)
    
    if mu_max - mu_min < 1e-10:
        mu_norm = np.zeros_like(mu)
    else:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
        
    if sig_max - sig_min < 1e-10:
        sig_norm = np.zeros_like(sigma_safe)
    else:
        sig_norm = (sigma_safe - sig_min) / (sig_max - sig_min)
        
    # Combine into a feature vector (2D)
    # We weight mu more heavily as it indicates location, sigma indicates uncertainty
    X = np.column_stack((mu_norm, 0.5 * sig_norm))
    
    # Compute pairwise squared distances
    # N=2000, so 2000x2000 matrix is manageable (4M floats ~ 32MB)
    # dist_sq[i, j] = ||X_i - X_j||^2
    X_norm_sq = np.sum(X**2, axis=1)
    dist_sq = X_norm_sq[:, np.newaxis] + X_norm_sq[np.newaxis, :] - 2 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0) # Numerical stability
    
    # Kernel bandwidth. 
    # A smaller bandwidth means less correlation, closer to pointwise EI.
    # A larger bandwidth means more correlation, stronger diversity penalty.
    # Let's use a length scale that covers ~10% of the feature space range.
    length_scale = 0.15
    rho = np.exp(-dist_sq / (2 * length_scale**2))
    
    # Diagonal is 1 (self-correlation)
    # For J-EI, we need to account for the joint distribution of improvements.
    # For a Gaussian process, the joint distribution of improvements I_i and I_j
    # depends on the covariance of the underlying function values.
    # Approximation: The expected joint improvement is roughly:
    # E[I_i + I_j] = EI_i + EI_j - Cov(I_i, I_j) * penalty_factor
    # A simpler proxy for diversity in greedy selection is to penalize candidates
    # that are highly correlated with already selected ones.
    
    # Greedy Selection of K candidates
    K = 20  # Batch size for diversity
    selected_indices = []
    selected_scores = np.zeros(len(mu))
    remaining = np.arange(len(mu))
    
    # We want to maximize total "value" which is base_score adjusted for redundancy.
    # Value of adding j to set S:
    # Delta_V(j, S) = base_score[j] - redundancy(j, S)
    # redundancy(j, S) = max_{i in S} (base_score[i] * rho[j, i]) ?
    # Or simply: sum_{i in S} rho[j, i] * base_score[i]
    
    # Let's use a standard greedy diversity maximization with value weighting.
    # Score(j) = base_score[j] * (1 - max_{i in S} rho[j, i])
    # This ensures we don't pick points too close to already picked high-value points.
    
    max_rho_with_selected = np.zeros(len(mu))
    
    for _ in range(K):
        if not remaining:
            break
            
        # Calculate adjusted score for remaining candidates
        # adjusted_score[j] = base_score[j] * (1 - max_rho_with_selected[j])
        # Note: max_rho_with_selected is 0 for unselected points initially
        
        # We want to pick the one with highest adjusted score
        # But we also need to update max_rho for the next iteration
        
        # Efficient update:
        # For each j in remaining, current penalty is max_rho_with_selected[j]
        # adjusted = base_score[remaining] * (1 - max_rho_with_selected[remaining])
        
        adjusted = base_score[remaining] * (1.0 - max_rho_with_selected[remaining])
        
        # Find best
        best_idx_in_remaining = np.argmax(adjusted)
        best_idx = remaining[best_idx_in_remaining]
        
        selected_indices.append(best_idx)
        
        # Update max_rho_with_selected for all remaining points
        # rho[best_idx, remaining] gives correlation between best and each remaining
        new_rho = rho[best_idx, remaining]
        # Update only where new_rho > current max
        mask = new_rho > max_rho_with_selected[remaining]
        max_rho_with_selected[remaining] = np.where(mask, new_rho, max_rho_with_selected[remaining])
        
        # Remove best from remaining
        remaining = np.delete(remaining, best_idx_in_remaining)
        
        # Check deadline
        if _check_deadline():
            break
    
    # 4. Assign Scores
    # Selected points get a high score, others get their base score (or lower)
    # To make the evaluator pick the selected ones, we boost them significantly.
    # However, if we boost them too much, we might miss a slightly better point not in the batch.
    # Let's assign:
    # selected: base_score * 1.5 + epsilon
    # others: base_score * 0.8
    # This creates a clear hierarchy but preserves some signal from base_score.
    
    scores = base_score.copy()
    if selected_indices:
        # Boost selected points
        # We want the top selected point to be the highest, etc.
        # Simply multiplying by a constant might not preserve order if base_scores vary.
        # Let's add a large constant to selected points to ensure they are picked first.
        # But we must return finite scores.
        # Let's use: score = base_score + boost * indicator
        boost = np.max(base_score) * 2.0 + 1.0
        scores[selected_indices] += boost
        
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
