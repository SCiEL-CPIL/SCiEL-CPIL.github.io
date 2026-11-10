# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses a two-stage approach:
    1. Initialize with Fibonacci sphere (good spread)
    2. Optimize using gradient-free methods (Nelder-Mead style) with time budget
    
    Returns:
        points: np.ndarray of shape (14, 3)
    """
    n = 14
    d = 3
    deadline = time.time() + 580  # safety margin before 630s limit
    
    def compute_score(points):
        """Compute (d_min/d_max)^2, return 0 for degenerate cases."""
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        if dists.size == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        if d_min < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2
    
    def fibonacci_sphere(n):
        """Generate n points on unit sphere using Fibonacci lattice."""
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        pts = np.column_stack((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ))
        return pts
    
    def nelder_mead_optimize(points, score_fn, max_iter=2000, timeout=570):
        """Simple Nelder-Mead optimization for the score function."""
        n_params = points.size
        # Initial simplex
        simplex = [points.flatten().copy() for _ in range(n_params + 1)]
        for i in range(n_params):
            simplex[i + 1] = simplex[0].copy()
            simplex[i + 1][i] += 0.1
        
        scores = [score_fn(s.reshape(n, d)) for s in simplex]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        start_time = time.time()
        for iteration in range(max_iter):
            if time.time() > start_time + timeout:
                break
            
            # Sort simplex by score (descending, since we want to maximize)
            order = np.argsort(scores)[::-1]
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_score = scores[-1]
            best_score = scores[0]
            
            # Reflection
            reflected = centroid + alpha * (centroid - worst)
            reflected_score = score_fn(reflected.reshape(n, d))
            
            if reflected_score >= best_score:
                # Try expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded_score = score_fn(expanded.reshape(n, d))
                if expanded_score > reflected_score:
                    simplex[-1] = expanded
                    scores[-1] = expanded_score
                else:
                    simplex[-1] = reflected
                    scores[-1] = reflected_score
            elif reflected_score >= scores[-2]:
                simplex[-1] = reflected
                scores[-1] = reflected_score
            else:
                # Contraction
                contracted = centroid + rho * (worst - centroid)
                contracted_score = score_fn(contracted.reshape(n, d))
                if contracted_score > worst_score:
                    simplex[-1] = contracted
                    scores[-1] = contracted_score
                else:
                    # Shrink
                    for i in range(1, n_params + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        scores[i] = score_fn(simplex[i].reshape(n, d))
        
        # Return best point
        best_idx = np.argmax(scores)
        return simplex[best_idx].reshape(n, d)
    
    # Stage 1: Fibonacci sphere initialization
    points = fibonacci_sphere(n)
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Stage 2: Optimize with Nelder-Mead
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        points = nelder_mead_optimize(points, compute_score, max_iter=5000, timeout=remaining_time - 5)
        score = compute_score(points)
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    # Stage 3: Try a few random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        rng = np.random.RandomState(123)
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Random perturbation of best points
            noise_scale = 0.05 + 0.1 * (1 - restart / 20.0)
            perturbed = best_points + noise_scale * rng.randn(n, d)
            
            # Quick local optimization
            quick_timeout = min(10, (deadline - time.time()) / 2)
            if quick_timeout < 2:
                break
            result = nelder_mead_optimize(perturbed, compute_score, max_iter=500, timeout=quick_timeout)
            result_score = compute_score(result)
            
            if result_score > best_score:
                best_points = result.copy()
                best_score = result_score
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    return best_points


# EVOLVE-BLOCK-END
