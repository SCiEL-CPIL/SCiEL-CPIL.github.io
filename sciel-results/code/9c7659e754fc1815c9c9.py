import numpy as np
import time
import sys

N = 130

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    
    Strategy: Riemannian Stochastic Gradient Descent (RSGD) with Dynamic Worst-Case Focus.
    1. Initialize with a Fibonacci spiral.
    2. Iteratively:
       a. Estimate the current covering radius by sampling a dense set of test points.
       b. Identify the "worst" test points (those farthest from any configuration point).
       c. Apply a repulsive gradient to the nearest configuration points, pushing them 
          away from the "worst" regions to reduce the maximum angular distance.
       d. Use an annealing schedule for step size to balance exploration and exploitation.
       e. Periodically re-sample the grid or increase its density as the solution converges.
    3. Return the best configuration found within the time limit.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_covering_radius = np.inf
    
    # Optimization Parameters
    # We use a stochastic approach: sample a subset of "test" points each iteration
    # to approximate the covering radius and gradient.
    
    # Initial grid size for test points. We start moderate and can increase if time allows.
    # A grid of ~5000-10000 points is a good balance for N=130.
    num_test_points = 5000
    
    # Generate a fixed set of random test points on the sphere for consistent evaluation
    # Using random points is faster than generating a grid each time, and covers the sphere uniformly.
    # We'll use a static set to avoid noise from changing the test set, but maybe regenerate
    # or increase density later? For now, a large static random set is a good proxy for the true max.
    # Actually, to be more robust, let's use a quasi-random sequence (Halton or similar) or just 
    # a large random set. Random is fine if N_test is large enough.
    
    rng = np.random.default_rng(42)
    # Generate random points on sphere
    test_points = rng.normal(size=(num_test_points, 3))
    test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    # Adaptive step size
    step_size = 0.05
    min_step_size = 1e-5
    cooling_rate = 0.999
    
    # Track the worst points to focus on
    # We will compute the distance from each test point to the nearest configuration point.
    # dist[i, j] = ||test_i - config_j||
    
    def compute_covering_metrics(pts):
        """
        Computes the covering radius estimate and identifies the worst test points.
        Returns:
        - max_dist: The maximum angular distance (in radians) from any test point to nearest config point.
        - worst_indices: Indices of test points that are at or near the maximum distance.
        - nearest_indices: For each worst test point, the index of the nearest config point.
        """
        # pts: (N, 3)
        # test_points: (M, 3)
        
        # Compute pairwise distances
        # diff = test_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # This is (M, N, 3). M=5000, N=130 -> 650,000 vectors. Memory: 650k * 3 * 8 bytes ~ 15MB. OK.
        
        # Squared distances
        # dist_sq = np.sum((test_points[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2)
        
        # More efficient: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        # Since all are unit vectors: ||a-b||^2 = 2 - 2 a.b
        # cos(theta) = a.b
        # theta = arccos(a.b)
        
        # We want to maximize theta, which is equivalent to minimizing cos(theta).
        # So we find the test point with the minimum dot product with any config point.
        
        dot_products = test_points @ pts.T  # (M, N)
        
        # For each test point i, find the nearest config point j (max dot product)
        max_dots = np.max(dot_products, axis=1)  # (M,)
        nearest_indices = np.argmax(dot_products, axis=1)  # (M,)
        
        # The angular distance is arccos(max_dot).
        # To avoid numerical issues with arccos, we can work with cos directly.
        # Max angular distance corresponds to min max_dot.
        
        min_max_dot = np.min(max_dots)
        
        # Identify worst points: those where max_dot is close to min_max_dot
        # Use a threshold to select a set of "worst" points for the gradient step.
        # Threshold: min_max_dot + small_epsilon
        threshold = min_max_dot + 1e-6
        worst_mask = max_dots <= threshold
        worst_indices = np.where(worst_mask)[0]
        
        # If too many or too few, adjust?
        # If only 1 point is worst, we focus on it. If many, we average the forces.
        
        # Calculate the actual covering radius in degrees for tracking
        # max_angle = np.arccos(np.clip(max_dots, -1, 1))
        # max_angle_deg = np.max(max_angle) * 180 / np.pi
        
        # For the gradient, we want to push the config points away from the worst test points.
        # The force on config point j from test point i is proportional to (test_i - config_j) * weight.
        # We only apply force from the "worst" test points to their nearest config points.
        
        return min_max_dot, worst_indices, nearest_indices, max_dots

    # Main Optimization Loop
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Compute metrics
        min_max_dot, worst_indices, nearest_indices, max_dots = compute_covering_metrics(points)
        
        # Calculate current covering radius estimate (in degrees) for tracking best
        # max_angle_rad = np.arccos(np.clip(min_max_dot, -1, 1))
        # current_radius_deg = max_angle_rad * 180 / np.pi
        # Instead of arccos every time, we can just track min_max_dot. Lower is better.
        
        if min_max_dot < best_covering_radius:
            best_covering_radius = min_max_dot
            best_points = points.copy()
        
        # If no worst points found (should not happen), break
        if len(worst_indices) == 0:
            break
            
        # Compute gradient (force) for each config point
        # Force on config point j is sum over worst test points i where nearest[i] == j of:
        #   (test_i - config_j) * weight
        # We use a weight that might depend on the "badness" (how far the test point is).
        # Let's use a simple constant weight for now, or weight by (1 - max_dot[i]) to emphasize the worst.
        
        forces = np.zeros_like(points)
        
        # Vectorized accumulation of forces
        # For each worst test point i:
        #   j = nearest_indices[i]
        #   force_j += (test_points[i] - points[j])
        
        # We can use np.add.at for efficient accumulation
        worst_test_points = test_points[worst_indices]
        worst_nearest_indices = nearest_indices[worst_indices]
        
        # Weight: emphasize points that are very far (low dot product)
        # weight = 1.0 + (1.0 - max_dots[worst_indices]) # Scale by badness
        
        # Simple accumulation
        # np.add.at(forces, worst_nearest_indices, worst_test_points)
        # But we need to subtract points[j] as well.
        
        # forces[j] += sum_{i: nearest[i]==j} (test_i - points[j])
        # = sum_{i: nearest[i]==j} test_i - points[j] * count_j
        
        sum_test = np.zeros_like(points)
        counts = np.zeros(N)
        
        np.add.at(sum_test, worst_nearest_indices, worst_test_points)
        np.add.at(counts, worst_nearest_indices, 1)
        
        # Avoid division by zero
        mask = counts > 0
        forces[mask] = sum_test[mask] - points[mask] * counts[mask, np.newaxis]
        
        # Normalize forces to unit length? Or scale by step size?
        # The magnitude of forces depends on the number of worst points.
        # Let's normalize each force vector to unit length before scaling by step size.
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Project force onto tangent plane
        dot = np.sum(forces * points, axis=1, keepdims=True)
        forces_tangent = forces - dot * points
        
        # Update points
        new_points = points + step_size * forces_tangent
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check for validity
        if not np.all(np.isfinite(new_points)):
            break
            
        points = new_points
        
        # Cool down step size
        step_size *= cooling_rate
        if step_size < min_step_size:
            # Try a small random kick to escape local minima?
            # Or just stop.
            # Let's try a few more iterations with a very small step or break.
            # Given the time limit, breaking is safer.
            # But maybe we can restart with a fresh step size?
            # For simplicity, break.
            # Actually, let's allow a few more iterations with min_step_size to settle.
            if iteration % 100 == 0:
                # Check time
                if time.time() - start_time > time_limit - 5:
                    break
            else:
                # Continue with min step for a while
                pass
        
        # Optional: Increase test point density if we have time and are stuck?
        # For now, keep it simple.
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    
    # Calculate final covering radius for reporting
    # Use a larger set of test points for final evaluation
    rng_final = np.random.default_rng(123)
    test_points_final = rng_final.normal(size=(20000, 3))
    test_points_final /= np.linalg.norm(test_points_final, axis=1, keepdims=True)
    
    dot_products = test_points_final @ pts.T
    max_dots = np.max(dot_products, axis=1)
    min_max_dot = np.min(max_dots)
    max_angle_rad = np.arccos(np.clip(min_max_dot, -1, 1))
    max_angle_deg = max_angle_rad * 180 / np.pi
    print(f"Estimated Covering Radius: {max_angle_deg:.6f} degrees")

