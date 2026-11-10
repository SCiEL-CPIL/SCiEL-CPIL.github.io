# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing to explore the configuration space, which is distinct from
    gradient-based methods. The points are constrained to the unit sphere.
    """
    n = 14
    d = 3
    
    best_score = 0.0
    best_points = None
    
    # Parameters for Simulated Annealing
    n_restarts = 20
    n_iterations = 5000
    T_start = 0.1
    T_end = 1e-6
    cooling_factor = (T_end / T_start) ** (1.0 / n_iterations)
    
    # Pre-compute indices for pdist to be faster if needed, but pdist is C-optimized.
    
    for r in range(n_restarts):
        # Initialize points on the unit sphere
        # Use a different initialization strategy per restart to diversify
        # Restart 0: Uniform random
        # Restart 1+: Random with slight structure or different seeds
        
        np.random.seed(42 + r)
        
        # Initialize
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        current_score = 0.0
        # Calculate initial score
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max > 0:
            current_score = d_min / d_max
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = T_start
        
        for i in range(n_iterations):
            # Generate a new candidate by perturbing one point
            idx = np.random.randint(n)
            # Perturb amount can be scaled by T or fixed
            # Fixed small perturbation is often more stable for SA on sphere
            step_size = 0.1 
            new_point = points[idx] + np.random.randn(d) * step_size
            new_norm = np.linalg.norm(new_point)
            if new_norm == 0:
                continue
            new_point /= new_norm
            
            # Store original to revert if needed
            old_point = points[idx].copy()
            points[idx] = new_point
            
            # Calculate new score
            dists = pdist(points)
            new_d_min = np.min(dists)
            new_d_max = np.max(dists)
            
            if new_d_max == 0:
                new_score = 0.0
            else:
                new_score = new_d_min / new_d_max
            
            # Acceptance criterion
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Probability of accepting worse solution
                # delta_score = new_score - current_score (negative)
                # P = exp(delta_score / T)
                delta = new_score - current_score
                if delta < 0:
                    # Avoid overflow/underflow issues, though T is small
                    if T > 1e-10:
                        prob = np.exp(delta / T)
                    else:
                        prob = 0.0
                    
                    if np.random.rand() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        points[idx] = old_point
                        # Keep current_score as is (the old one)
                
            # Cool down
            T *= cooling_factor
            
        # After SA, do a few gradient steps to polish? 
        # The directive says "do not switch to a different design". 
        # However, combining SA with a local polish is still "SA approach" primarily.
        # But to be safe and strictly follow "new approach", I will stick to pure SA 
        # or just rely on the high iteration count of SA.
        
    if best_points is None:
        # Fallback
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
