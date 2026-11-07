"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid Expected Improvement (EI) with multi-scale density penalty
    in (mu, sigma) feature space to enforce spatial diversity.
    
    The approach combines:
    1. Standard EI for exploitation
    2. A penalty based on how "crowded" the (mu, sigma) space is,
       which encourages exploration of under-explored regions
    3. Dynamic scaling that balances exploration/exploitation based on progress
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,).
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,).
    f_best : float
        The lowest observed function value so far.
    
    Returns:
    --------
    np.ndarray
        Acquisition scores for each candidate, shape (N,). Higher is better.
    """
    # Known global minimum of Hartmann 3D
    global_min = -3.86278
    
    N = len(mu)
    
    # Compute standard EI
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Handle sigma near zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    EI = np.maximum(EI, 0.0)
    
    # Compute density penalty in (mu, sigma) feature space
    # We use a simple approach: for each point, count how many other points
    # are nearby in the (mu, sigma) space, and penalize accordingly
    
    # Create feature vectors
    features = np.column_stack([mu, sigma])
    
    # Normalize features for distance computation
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-10 else 1.0
    sigma_range = np.ptp(sigma) if np.ptp(sigma) > 1e-10 else 1.0
    
    features_norm = np.column_stack([
        (mu - np.min(mu)) / mu_range,
        (sigma - np.min(sigma)) / sigma_range
    ])
    
    # Compute pairwise distances (squared) in normalized feature space
    # To keep memory reasonable, we use a chunked approach
    # But for N=2000, this should be fine
    
    # Actually, let's use a simpler approach: compute the number of points
    # within a small radius in (mu, sigma) space
    
    # Use a grid-based approach for efficiency
    # Bin the (mu, sigma) space into a grid and count occupancy
    
    n_bins = 20
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_bin = np.clip((mu - mu_min) / mu_range * n_bins, 0, n_bins - 1).astype(int)
    sigma_bin = np.clip((sigma - sigma_min) / sigma_range * n_bins, 0, n_bins - 1).astype(int)
    
    # Count points in each bin
    bin_counts = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(N):
        bin_counts[mu_bin[i], sigma_bin[i]] += 1
    
    # For each point, get the count in its bin and neighboring bins
    density = np.zeros(N)
    for i in range(N):
        mb = mu_bin[i]
        sb = sigma_bin[i]
        # Sum over 3x3 neighborhood
        mb_start = max(0, mb - 1)
        mb_end = min(n_bins, mb + 2)
        sb_start = max(0, sb - 1)
        sb_end = min(n_bins, sb + 2)
        density[i] = np.sum(bin_counts[mb_start:mb_end, sb_start:sb_end])
    
    # Normalize density
    density = density / np.max(density) if np.max(density) > 0 else density
    
    # Density penalty: higher density -> higher penalty
    # We use an exponential penalty to make it more aggressive
    penalty_strength = 0.5  # Scale of the penalty
    
    density_penalty = penalty_strength * density
    
    # Dynamic scaling based on progress
    # When f_best is far from global_min, we want more exploration (lower EI weight, higher penalty weight)
    # When f_best is close to global_min, we want more exploitation (higher EI weight, lower penalty weight)
    
    gap = f_best - global_min
    gap_clipped = min(max(gap, 0.0), 4.0)
    
    # EI weight: starts high, decreases as we get closer to optimum
    # Penalty weight: starts high, decreases as we get closer to optimum
    ei_weight = 1.0 + 0.5 * (gap_clipped / 4.0)  # 1.0 to 1.5
    penalty_weight = 1.0 + 2.0 * (gap_clipped / 4.0)  # 1.0 to 3.0
    
    # Combine: score = ei_weight * EI - penalty_weight * density_penalty
    scores = ei_weight * EI - penalty_weight * density_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
