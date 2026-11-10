# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 600.0
    n = 16
    
    # Function to compute the score R = (d_min / d_max)^2
    def compute_score(points):
        if points.shape != (n, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Soft objective for gradient calculation
    # Maximize log(d_min) - log(d_max)
    # Approximate with soft-min and soft-max
    def soft_objective(points, k):
        d = pdist(points)
        # To avoid log(0) and numerical issues, add a small epsilon or rely on k
        # log-sum-exp trick for stability
        # S_min = -log(sum(exp(-d/k)))
        # S_max = log(sum(exp(d/k)))
        # We want to maximize S_min - S_max
        
        # Scale distances to avoid overflow
        # d is typically O(1)
        
        # Calculate gradients with respect to points
        # This is computationally expensive if done naively for every step.
        # Let's use a finite difference or analytic gradient for the soft objective.
        
        # Actually, for a robust and fast solution within 600s, 
        # a coordinate ascent / local search on the exact objective might be faster 
        # and more reliable than gradient ascent on a soft approximation 
        # which requires tuning k and step sizes.
        
        # Let's use a hybrid: 
        # 1. Initialize with regular 16-gon.
        # 2. Use a local search that moves points to increase the minimum distance 
        #    while keeping the maximum distance fixed (or minimizing it).
        #    Actually, maximizing d_min/d_max is equivalent to 
        #    maximizing d_min subject to d_max <= 1.
        
        # Algorithm:
        # 1. Normalize points so d_max = 1.
        # 2. Maximize d_min.
        # 3. Repeat with different initializations.
        
        pass

    def normalize_points(points):
        d = pdist(points)
        d_max = np.max(d)
        if d_max == 0:
            return points
        return points / d_max

    def local_search(points, max_iters=5000, tol=1e-8):
        """
        Local search to maximize d_min subject to d_max <= 1.
        We use a gradient-free approach: 
        For each point, try to move it to increase the minimum distance to its neighbors,
        while ensuring no distance exceeds 1.
        """
        points = points.copy()
        points = normalize_points(points)
        
        # Precompute pairs
        # We will iterate over points and try to move them.
        
        current_d_min = np.min(pdist(points))
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            improved = False
            # Try to move each point
            for i in range(n):
                # Calculate distances from point i to all others
                d_i = np.linalg.norm(points - points[i], axis=1)
                d_i[i] = np.inf # Exclude self
                
                # Identify the closest neighbor
                j_min = np.argmin(d_i)
                d_min_i = d_i[j_min]
                
                # Identify the farthest neighbor (to check constraint)
                j_max = np.argmax(d_i)
                d_max_i = d_i[j_max]
                
                # We want to move point i away from j_min.
                # Direction: from j_min to i
                dir_vec = points[i] - points[j_min]
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir < 1e-10:
                    continue
                dir_vec = dir_vec / norm_dir
                
                # Try a small step in this direction
                # Step size should be small enough not to violate d_max <= 1
                # Current max distance involving i is d_max_i.
                # If we move i, distances to all other points change.
                # A safe step size is limited by (1 - d_max_i) / 2 roughly?
                # Let's try a small step and check.
                
                step_sizes = [1e-4, 1e-3, 1e-2, 1e-1]
                best_new_points = None
                best_new_d_min = current_d_min
                
                for ss in step_sizes:
                    new_points = points.copy()
                    new_points[i] = points[i] + ss * dir_vec
                    
                    # Check if any distance exceeds 1
                    d_new = pdist(new_points)
                    if np.max(d_new) > 1.0 + 1e-9:
                        continue
                    
                    new_d_min = np.min(d_new)
                    if new_d_min > best_new_d_min + tol:
                        best_new_d_min = new_d_min
                        best_new_points = new_points
                        improved = True
                
                if best_new_points is not None:
                    points = best_new_points
                    current_d_min = best_new_d_min
                    # Re-normalize? No, we are maintaining d_max <= 1.
                    # But moving points might make d_max < 1.
                    # We should re-normalize to keep d_max = 1 to maximize the scale?
                    # Actually, if d_max < 1, we can scale up to make d_max = 1, 
                    # which increases d_min.
                    # So, after any move, we should check if we can scale up.
                    d_max_new = np.max(pdist(points))
                    if d_max_new < 1.0 - 1e-9:
                        points = points / d_max_new
                        current_d_min = np.min(pdist(points))
                        improved = True

            if not improved:
                break
                
        return points

    # Generate initializations
    inits = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_16 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(regular_16)
    
    # 2. Perturbed regular 16-gons
    rng = np.random.RandomState(42)
    for _ in range(20):
        noise = rng.randn(n, 2) * 0.1
        inits.append(regular_16 + noise)
        
    # 3. Random points (for diversity, though likely poor)
    # for _ in range(5):
    #     inits.append(rng.randn(n, 2))

    best_points = None
    best_score = -np.inf

    for init in inits:
        if time.time() > deadline - 60:
            break
            
        # Run local search
        candidate = local_search(init, max_iters=2000)
        
        # Final normalization to ensure d_max = 1 (or just compute score)
        score = compute_score(candidate)
        
        if score > best_score:
            best_score = score
            best_points = candidate

    return best_points

# EVOLVE-BLOCK-END
