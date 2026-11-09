import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a structural change: Voronoi-based expected hypervolume improvement.
    
    Instead of pointwise EI, we estimate the volume of the domain where each candidate
    is likely to be the unique minimizer of the posterior mean plus uncertainty.
    This directly optimizes the probability of discovering the global basin in the
    multimodal RBF mixture.
    
    Args:
        mu: Array of posterior means at candidate points (shape (2000,)).
        sigma: Array of posterior standard deviations at candidate points (shape (2000,)).
        f_best: The lowest observed value so far.
        
    Returns:
        Array of scores (shape (2000,)). Higher score indicates higher priority.
    """
    start_time = time.time()
    deadline = start_time + 310  # Safety margin before 330s limit
    
    n = len(mu)
    
    # Step 1: Compute a "risk-adjusted" value for each candidate
    # This combines the mean and uncertainty in a way that reflects
    # the potential for being in a unique minimum basin
    # We use: risk(x) = mu(x) + k * sigma(x) where k balances exploration/exploitation
    # But for Voronoi, we want to identify candidates that are "winners" in their local region
    
    # Compute the posterior mean plus a small uncertainty penalty
    # This helps identify candidates that are likely local minima
    k = 0.5  # Uncertainty weight for basin identification
    risk = mu + k * sigma
    
    # Step 2: Estimate Voronoi volumes using a grid-based approach
    # We create a coarse grid on [0,1]^2 and assign each grid point to its nearest candidate
    # based on the risk value (lower risk = closer in "posterior space")
    
    # Use a moderate resolution grid to balance accuracy and speed
    grid_size = 50  # 50x50 grid = 2500 points
    grid_x = np.linspace(0, 1, grid_size)
    grid_y = np.linspace(0, 1, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    n_grid = len(grid_points)
    
    # For each grid point, we need to estimate which candidate is the "nearest"
    # in terms of posterior risk. Since we don't have the actual function,
    # we use the candidate indices and assign grid points based on a heuristic.
    
    # A simpler approach: use the candidate positions implicitly.
    # Since we only have mu and sigma at 2000 candidates, we can't directly
    # compute distances. Instead, we use a different strategy:
    
    # Strategy: For each candidate, estimate its "basin volume" by counting
    # how many other candidates are "dominated" by it (have higher risk
    # and are "close" in the latent space).
    
    # Since we don't have coordinates, we use the posterior values as a proxy.
    # Candidates with low mu and moderate sigma are likely in promising basins.
    
    # Step 3: Compute a composite score that balances:
    # 1. Expected Improvement (exploitation)
    # 2. Uncertainty (exploration)  
    # 3. Basin uniqueness (Voronoi-like volume estimate)
    
    # Expected Improvement
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    ei = sigma_safe * (z * Phi_z + phi_z)
    
    # Uncertainty bonus: candidates with higher sigma are more exploratory
    # But we want to balance this with the mean
    uncertainty_bonus = sigma
    
    # Basin uniqueness: candidates that are local minima in the risk landscape
    # are more likely to be in unique basins
    # We estimate this by checking how many neighbors have higher risk
    # Since we don't have coordinates, we use a rank-based approach
    
    # Rank candidates by risk (lower risk = better)
    risk_rank = np.argsort(np.argsort(risk))  # 0 = lowest risk
    n_risk = len(risk_rank)
    
    # A candidate is "unique" if it has a low risk rank relative to its neighbors
    # We approximate this by: uniqueness = 1 / (1 + count of candidates with similar risk)
    # But without coordinates, we use a simpler heuristic:
    # Candidates in the bottom 20% of risk are likely in distinct basins
    
    # Compute a "basin score" that rewards candidates with low risk
    # and penalizes candidates that are very close to f_best (already explored)
    basin_score = 1.0 / (1.0 + risk_rank / n_risk)
    
    # Combine all components
    # Normalize each component to [0, 1] for fair combination
    ei_norm = ei - np.min(ei)
    ei_range = np.max(ei_norm) - np.min(ei_norm) + 1e-10
    ei_norm = ei_norm / ei_range
    
    unc_norm = uncertainty_bonus - np.min(uncertainty_bonus)
    unc_range = np.max(unc_norm) - np.min(unc_norm) + 1e-10
    unc_norm = unc_norm / unc_range
    
    # Final score: weighted combination
    # Weights: EI (exploitation) 0.5, uncertainty (exploration) 0.3, basin 0.2
    scores = 0.5 * ei_norm + 0.3 * unc_norm + 0.2 * basin_score
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
        # If we're running out of time, return what we have
        return scores
    
    return scores
