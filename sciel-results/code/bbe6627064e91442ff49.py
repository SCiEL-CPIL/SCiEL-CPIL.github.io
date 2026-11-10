# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3
    deadline = time.time() + 620  # 620s to leave 10s margin

    best_score = -1.0
    best_points = np.zeros((n, d))

    # Helper to calculate score based on squared distances
    def get_score(pts):
        # Calculate squared pairwise distances
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        if d_max_sq <= 0:
            return 0.0
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    # Simulated annealing
    def anneal(initial_points, max_iters=100000, start_temp=1.0, end_temp=0.001):
        nonlocal best_score, best_points
        
        current_points = initial_points.copy()
        current_score = get_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            perturbation_scale = 0.1 * temp
            
            # Random perturbation
            idx = np.random.randint(n)
            perturbation = np.random.randn(d) * perturbation_scale
            new_points = current_points.copy()
            new_points[idx] += perturbation
            
            new_score = get_score(new_points)
            
            # Accept or reject
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / temp):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        return best_points

    # Try multiple random starts
    np.random.seed(42)
    n_starts = 50
    max_iters_per_start = 5000
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        points = np.random.randn(n, d)
        
        # Run annealing
        anneal(points, max_iters=max_iters_per_start)
        
        # Reduce iterations for later starts to save time
        if start_idx > 20:
            max_iters_per_start = 2000
        elif start_idx > 30:
            max_iters_per_start = 1000
    
    # Final refinement with more iterations
    if time.time() < deadline - 60:
        anneal(best_points, max_iters=50000, start_temp=0.1, end_temp=0.0001)

    return best_points


# EVOLVE-BLOCK-END
