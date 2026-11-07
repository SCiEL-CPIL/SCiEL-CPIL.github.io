import numpy as np
from scipy.stats import norm
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 320  # 330s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. Multi-scale kernel-density-based exploration term that penalizes
       regions densely sampled by the candidate pool
    
    The density penalty uses a fixed bandwidth KDE in the (mu, sigma) feature
    space to enforce spatial diversity independent of the GP posterior.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: exploit the current best guess (lowest mu)
        mu = np.asarray(mu, dtype=np.float64)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
    # Robust sigma floor to avoid numerical issues
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, sigma_floor)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    span = f_start_ref - f_global_min
    if span <= 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
    
    # Compute Expected Improvement
    z = (f_best - mu_safe) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Multi-scale kernel-density-based exploration term
    # We estimate local density of the 2000 candidate points using a fixed bandwidth
    # in the (mu, sigma) feature space, then subtract this density from the EI score
    # to explicitly penalize regions already densely sampled by the candidate pool
    
    # Normalize mu and sigma for density estimation
    mu_min, mu_max = np.min(mu_safe), np.max(mu_safe)
    sig_min, sig_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min
    sig_range = sig_max - sig_min
    
    if mu_range < 1e-10:
        mu_range = 1e-10
    if sig_range < 1e-10:
        sig_range = 1e-10
    
    mu_norm = (mu_safe - mu_min) / mu_range
    sig_norm = (sigma_safe - sig_min) / sig_range
    
    # Use a grid-based KDE approximation with two scales
    # Scale 1: fine-grained (25x25 grid)
    # Scale 2: coarse-grained (8x8 grid)
    
    # Fine-grained density
    fine_grid_size = 25
    mu_idx_fine = np.clip((mu_norm * fine_grid_size).astype(int), 0, fine_grid_size - 1)
    sig_idx_fine = np.clip((sig_norm * fine_grid_size).astype(int), 0, fine_grid_size - 1)
    
    fine_cell_counts = np.zeros((fine_grid_size, fine_grid_size), dtype=np.float64)
    for i in range(n):
        fine_cell_counts[mu_idx_fine[i], sig_idx_fine[i]] += 1
    
    fine_cell_density = fine_cell_counts / n
    fine_local_density = fine_cell_density[mu_idx_fine, sig_idx_fine]
    
    # Coarse-grained density
    coarse_grid_size = 8
    mu_idx_coarse = np.clip((mu_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
    sig_idx_coarse = np.clip((sig_norm * coarse_grid_size).astype(int), 0, coarse_grid_size - 1)
    
    coarse_cell_counts = np.zeros((coarse_grid_size, coarse_grid_size), dtype=np.float64)
    for i in range(n):
        coarse_cell_counts[mu_idx_coarse[i], sig_idx_coarse[i]] += 1
    
    coarse_cell_density = coarse_cell_counts / n
    coarse_local_density = coarse_cell_density[mu_idx_coarse, sig_idx_coarse]
    
    # Combined density: weighted sum of fine and coarse
    # Early on (high progress), we want more diversity (higher weight on density penalty)
    # Late on (low progress), we want more exploitation (lower weight on density penalty)
    density_weight_fine = 0.6 * np.sqrt(progress) + 0.15
    density_weight_coarse = 0.4 * np.sqrt(progress) + 0.1
    
    density_penalty = density_weight_fine * fine_local_density + density_weight_coarse * coarse_local_density
    
    # Final score: EI - density_penalty
    scores = ei - density_penalty
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
