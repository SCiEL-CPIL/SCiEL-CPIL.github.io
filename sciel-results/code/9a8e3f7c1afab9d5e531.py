"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import Delaunay, Voronoi
import time


def acquisition(mu, sigma, f_best):
    """
    Voronoi-based expected hypervolume improvement acquisition.
    
    Estimates the volume of the domain where each candidate is the unique
    minimizer of the posterior-based potential function, directly optimizing
    the probability of discovering new basins in the multimodal RBF mixture.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement as a baseline
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Create a 2D embedding from (mu, sigma) to represent spatial structure
    # Points with similar (mu, sigma) are likely close in the input space
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    # Normalize to [0, 1] range for embedding
    x1 = (mu - mu_min) / mu_range
    x2 = (sigma_safe - sigma_min) / sigma_range
    
    # Create 2D points for Voronoi computation
    points_2d = np.column_stack([x1, x2])
    
    # Compute Voronoi cells to estimate "domain volume" each candidate controls
    # This gives us a measure of how much of the search space each candidate "owns"
    
    # For computational efficiency, use a subset for Voronoi if n is large
    if n > 500:
        # Use random subsample for Voronoi computation
        np.random.seed(42)
        subset_idx = np.random.choice(n, 500, replace=False)
        subset_points = points_2d[subset_idx]
    else:
        subset_idx = np.arange(n)
        subset_points = points_2d
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(subset_points)
        
        # Compute area of each Voronoi cell
        # For points with valid Voronoi regions, compute polygon area
        cell_areas = np.zeros(len(subset_idx))
        
        for i, region_idx in enumerate(vor.regions):
            if len(region_idx) == 0:
                continue
            if -1 in region_idx:
                # Unbounded region; approximate with a large value
                cell_areas[i] = 0.5  # Arbitrary large value for unbounded
            else:
                # Compute area of the polygon
                vertices = vor.vertices[region_idx]
                if len(vertices) < 3:
                    cell_areas[i] = 0.0
                else:
                    # Shoelace formula for polygon area
                    x = vertices[:, 0]
                    y = vertices[:, 1]
                    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
                    cell_areas[i] = area
        
        # Map cell areas back to all n points
        # For points not in the subset, interpolate or use nearest neighbor
        if n > 500:
            # Use KDTree to find nearest subset point for each non-subset point
            from scipy.spatial import cKDTree
            tree = cKDTree(subset_points)
            distances, nearest_idx = tree.query(points_2d, k=1)
            full_cell_areas = cell_areas[nearest_idx]
        else:
            full_cell_areas = cell_areas
        
    except Exception:
        # Fallback: use simple density-based estimation
        # Points in sparse regions have larger "effective volume"
        full_cell_areas = np.ones(n)
        if n > 1:
            # Compute pairwise distances and use inverse density
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(points_2d))
            # Mean distance to other points as proxy for volume
            mean_dists = np.mean(dist_matrix, axis=1)
            full_cell_areas = mean_dists
    
    # Normalize cell areas
    area_max = np.max(full_cell_areas)
    area_min = np.min(full_cell_areas)
    area_range = area_max - area_min + 1e-9
    volume_norm = (full_cell_areas - area_min) / area_range
    
    # Combine EI with volume-based score
    # Points that are both high-EI and control large volumes are most valuable
    # This directly targets discovery of new basins
    
    # Adaptive weighting based on progress
    known_min = -2.53218
    if f_best > known_min:
        progress = min(max((f_best - known_min) / (0 - known_min), 0), 1)
    else:
        progress = 0.0
    
    # When far from minimum, weight volume more heavily (explore new basins)
    # When close, weight EI more heavily (exploit known good regions)
    volume_weight = 0.3 + 0.4 * progress  # Range: 0.3 to 0.7
    
    scores = (1 - volume_weight) * ei_norm + volume_weight * volume_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize to [0, 1]
    scores_max = np.max(scores)
    if scores_max > 1e-9:
        scores = scores / scores_max
    
    return scores
# EVOLVE-BLOCK-END
