# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _rotate_random(points, rng):
    """
    Apply a random rotation to a random subset of points.
    This moves points on the sphere while preserving distances to the origin (if they were on the sphere).
    However, since we are not strictly on the sphere (we allow radial movement), 
    we actually want to perturb positions in 3D space.
    
    A better approach for 'shake':
    1. Pick a random point.
    2. Generate a random direction in 3D.
    3. Move the point by step_size * direction.
    4. Re-center and re-scale the whole configuration.
    """
    n = len(points)
    idx = rng.integers(0, n)
    # Random direction
    dir = rng.normal(size=3)
    dir = dir / np.linalg.norm(dir)
    
    # Move the point
    new_points = points.copy()
    new_points[idx] += dir * 1.0 # Step size handled externally or fixed here? 
    # Let's use a fixed step size in the caller, or pass it.
    # For simplicity, let's make this function take a step size.
    return new_points

def _recenter_rescale(points):
    """Center at origin and scale so that sum(||p_i||^2) = 1."""
    p = points - np.mean(points, axis=0)
    norm_sq = np.sum(np.sum(p**2, axis=1))
    if norm_sq < 1e-12:
        return points
    p = p / np.sqrt(norm_sq)
    return p

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 14 points in 3D.
    Strategy: Stochastic geometric perturbation with annealing.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    n = 14
    best_score = 0.0
    best_points = None
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # Initialize with Fibonacci sphere
    current_points = fibonacci_sphere(n)
    current_points = _recenter_rescale(current_points)
    current_score = _score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    # Annealing parameters
    # Start with large steps to explore, end with small steps to refine
    initial_step = 0.5
    final_step = 0.001
    total_iterations = 50000  # We will break based on time
    
    # We will run multiple "restarts" if we stall, or just continuous annealing.
    # Continuous annealing is usually better for this type of landscape.
    
    for i in range(total_iterations):
        if time.time() > deadline:
            break
            
        # Linear annealing of step size
        progress = i / total_iterations
        step_size = initial_step * (1 - progress) + final_step * progress
        
        # Generate a perturbed version
        # Strategy: Perturb 1-3 random points with random 3D vectors
        num_perturbed = rng.integers(1, 4)
        indices = rng.choice(n, size=num_perturbed, replace=False)
        
        new_points = current_points.copy()
        for idx in indices:
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            new_points[idx] += step_size * direction
            
        # Re-center and re-scale
        new_points = _recenter_rescale(new_points)
        
        # Evaluate
        new_score = _score(new_points)
        
        # Accept if better, or with some probability if worse (Simulated Annealing style)
        # But since we have a hard time limit and no temperature schedule that's easy to tune,
        # let's use a "Hill Climbing with Random Restarts" or "Accept if not too much worse"
        # Actually, pure hill climbing gets stuck. 
        # Let's try: Accept if new_score > current_score.
        # If we get stuck (no improvement for X iterations), do a larger random jump.
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            stuck_counter = 0
        else:
            # Stuck logic
            # We don't have a counter in the loop scope easily without passing state.
            # Let's just rely on the annealing to shrink steps.
            # If step size is very small and we are stuck, we might be in a local min.
            # To escape, we can occasionally do a big random jump.
            if step_size < 0.01 and rng.random() < 0.01:
                # Big random jump: reset a few points to random positions
                jump_indices = rng.choice(n, size=3, replace=False)
                current_points[jump_indices] = rng.normal(size=(3, 3))
                current_points = _recenter_rescale(current_points)
                current_score = _score(current_points)
                # We don't update best here immediately, let the next iteration handle it
                # Or just let it be, it might be worse.
                
    return best_points

# EVOLVE-BLOCK-END
