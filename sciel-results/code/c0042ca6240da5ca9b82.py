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

def _local_polish(points, iterations=200, lr_start=0.01):
    """
    Perform a short gradient-based polish to settle the points 
    after a random kick. This helps recover from the noise of the kick.
    """
    current_points = points.copy()
    lr = lr_start
    
    for i in range(iterations):
        # Calculate repulsive forces (same as before, but used only for polish)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_matrix = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists_matrix[~np.isfinite(dists_matrix)] = 1e-6
        dists_matrix[np.diag_indices_from(dists_matrix)] = 1.0
        
        power = 10
        norm_p = dists_matrix ** (power + 1)
        norm_p[norm_p == 0] = 1e-10
        
        force_mat = diff / norm_p[:, :, np.newaxis]
        forces = np.sum(force_mat, axis=1)
        
        # Update
        current_points += lr * forces
        
        # Project to unit sphere
        norms = np.linalg.norm(current_points, axis=1, keepdims=True)
        current_points /= norms
        
        # Center
        current_points -= np.mean(current_points, axis=0, keepdims=True)
        norms = np.linalg.norm(current_points, axis=1, keepdims=True)
        current_points /= norms
        
        # Decay learning rate
        lr *= 0.95
        if lr < 1e-6:
            lr = 1e-6
            
    return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing with random kicks and local gradient polishing.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    best_score = 0.0
    best_points = None
    n = 14
    
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
    
    # Initial best is the fibonacci sphere
    init_points = fibonacci_sphere(n)
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Simulated Annealing Parameters
    # We want to maximize score, so we can think of "energy" as -score.
    # Or simply: accept if new_score > old_score, or with probability exp((new-old)/T).
    # Since score is positive, let's use log score for stability in the exponent.
    # Actually, score is (d_min/d_max)^2. 
    # Let's use the score directly. If new < old, accept with prob exp((new - old) / T).
    # Note: (new - old) will be negative.
    
    T_start = 0.1
    T_end = 1e-6
    num_sa_steps = 20000  # Adjust based on speed
    kick_magnitude = 0.2
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Track temperature schedule
    for step in range(num_sa_steps):
        if time.time() > deadline - 10:
            break
            
        # Cooling schedule: exponential decay
        alpha = step / num_sa_steps
        T = T_start * (T_end / T_start) ** alpha
        
        # 1. Generate a kick
        # Pick a random point and perturb it
        idx = np.random.randint(n)
        kick = np.random.randn(3) * kick_magnitude
        new_points = current_points.copy()
        new_points[idx] += kick
        
        # Project to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        new_points -= np.mean(new_points, axis=0, keepdims=True)
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points /= norms
        
        # 2. Evaluate new score
        new_score = _score(new_points)
        
        # 3. Acceptance criterion
        if new_score >= current_score:
            # Always accept improvement
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept worse solution with probability exp((new - old) / T)
            # To avoid underflow, we can use log
            # prob = exp((new_score - current_score) / T)
            # If T is very small, this is 0.
            delta = new_score - current_score
            # If delta is very negative, exp(delta/T) is 0.
            # We can just compute it.
            if T > 1e-9:
                accept_prob = np.exp(delta / T)
            else:
                accept_prob = 0.0
                
            if np.random.random() < accept_prob:
                current_points = new_points
                current_score = new_score
        
        # 4. Occasional local polish to settle the basin
        # Every 500 steps, do a quick polish on the current state
        if step % 500 == 0 and step > 0:
            polished_points = _local_polish(current_points, iterations=50, lr_start=0.005)
            polished_score = _score(polished_points)
            
            if polished_score > current_score:
                current_points = polished_points
                current_score = polished_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            # Also, if we are stuck (no improvement in last 1000 steps), 
            # we might want to kick harder or restart from best.
            # For simplicity, we just continue. The kick magnitude is fixed.
            # We could adapt kick magnitude, but let's keep it simple for now.
            
    # Final polish on the best found
    if best_points is not None:
        # Run a more extensive polish on the best configuration
        # Use a smaller learning rate and more iterations
        final_points = _local_polish(best_points, iterations=1000, lr_start=0.001)
        final_score = _score(final_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()

    # If time remains, do another round of SA starting from best?
    # Or just return best.
    # Given the 600s limit, we likely used most of it in the loop.
    
    if best_points is None:
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
