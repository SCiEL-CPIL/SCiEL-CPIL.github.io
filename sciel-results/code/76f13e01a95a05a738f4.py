import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with multiple restarts to find a near-optimal 
    packing of N circles in a unit circle.
    """
    
    def compute_score(points):
        """Compute the actual packing score for given points."""
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Check boundary
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii >= 1.0):
            return 0.0
        
        # Check pairwise distances
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        
        min_boundary = 1.0 - np.max(radii)
        return min(min_pair_dist / 2.0, min_boundary)
    
    def annealing_score(points, r_target):
        """
        Score for annealing: penalizes violations of radius r_target.
        Lower is better.
        """
        # Boundary violation
        radii = np.linalg.norm(points, axis=1)
        boundary_viol = np.maximum(0, radii + r_target - 1.0)
        boundary_pen = np.sum(boundary_viol ** 2)
        
        # Pairwise violation
        pair_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < 2 * r_target:
                    pair_viol += (2 * r_target - d) ** 2
        
        return boundary_pen + pair_viol
    
    def simulate_annealing(initial_points, r_target, start_temp=0.1, end_temp=1e-6, 
                           n_iterations=50000, time_limit=110.0):
        """
        Run simulated annealing to minimize violations for target radius r_target.
        Returns best points found.
        """
        current_points = initial_points.copy()
        current_score = annealing_score(current_points, r_target)
        best_points = current_points.copy()
        best_score = current_score
        
        start_time = time.time()
        
        for it in range(n_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Temperature schedule: exponential cooling
            progress = it / n_iterations
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Generate neighbor: perturb one or two points
            n_perturb = 1 if np.random.rand() < 0.7 else 2
            indices = np.random.choice(N, size=n_perturb, replace=False)
            
            neighbor_points = current_points.copy()
            for idx in indices:
                # Random perturbation in polar coordinates
                r = np.linalg.norm(neighbor_points[idx])
                theta = np.arctan2(neighbor_points[idx][1], neighbor_points[idx][0])
                
                delta_r = np.random.normal(0, 0.01)
                delta_theta = np.random.normal(0, 0.05)
                
                new_r = max(0, r + delta_r)
                new_theta = theta + delta_theta
                
                neighbor_points[idx] = [new_r * np.cos(new_theta), new_r * np.sin(new_theta)]
            
            # Clip to ensure points stay within reasonable bounds
            norms = np.linalg.norm(neighbor_points, axis=1)
            mask = norms > 1.5
            if np.any(mask):
                neighbor_points[mask] = neighbor_points[mask] / norms[mask][:, np.newaxis] * 1.5
            
            neighbor_score = annealing_score(neighbor_points, r_target)
            
            # Accept or reject
            if neighbor_score < current_score:
                current_points = neighbor_points
                current_score = neighbor_score
                if current_score < best_score:
                    best_points = current_points.copy()
                    best_score = current_score
            else:
                # Accept with probability exp(-delta/temp)
                delta = neighbor_score - current_score
                if delta < 1e-10:
                    prob = 1.0
                else:
                    prob = np.exp(-delta / max(temp, 1e-10))
                if np.random.rand() < prob:
                    current_points = neighbor_points
                    current_score = neighbor_score
        
        return best_points, best_score
    
    # Multiple restarts with different initial configurations
    start_time = time.time()
    time_limit = 115.0
    
    best_overall_points = None
    best_overall_score = 0.0
    
    # Try different initial configurations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Fermat spiral
            points = np.zeros((N, 2))
            for i in range(N):
                r_init = np.sqrt(i / N) * 0.9
                theta = i * np.pi * (3 - np.sqrt(5))
                points[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
        elif restart == 1:
            # Random uniform
            points = np.random.randn(N, 2)
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * np.random.uniform(0, 0.8, size=(N, 1))
        elif restart == 2:
            # Ring-based initialization
            points = np.zeros((N, 2))
            # Center point
            points[0] = [0, 0]
            # Inner ring (12 points)
            for i in range(1, 13):
                theta = (i - 1) * 2 * np.pi / 12
                points[i] = [0.3 * np.cos(theta), 0.3 * np.sin(theta)]
            # Middle ring (18 points)
            for i in range(13, 31):
                theta = (i - 13) * 2 * np.pi / 18
                points[i] = [0.55 * np.cos(theta), 0.55 * np.sin(theta)]
            # Outer ring (18 points)
            for i in range(31, 50):
                theta = (i - 31) * 2 * np.pi / 18
                points[i] = [0.8 * np.cos(theta), 0.8 * np.sin(theta)]
        elif restart == 3:
            # Hexagonal-ish grid
            points = np.zeros((N, 2))
            positions = []
            for row in range(-3, 4):
                for col in range(-3, 4):
                    x = col * 0.15 + (0.075 if row % 2 != 0 else 0)
                    y = row * 0.13
                    if np.sqrt(x*x + y*y) < 0.9:
                        positions.append([x, y])
            # Take first N
            for i in range(min(N, len(positions))):
                points[i] = positions[i]
            # Fill remaining randomly
            for i in range(len(positions), N):
                points[i] = np.random.randn(2) * 0.3
        else:
            # Random with different scales
            scale = 0.5 + 0.1 * restart
            points = np.random.randn(N, 2) * scale
            norms = np.linalg.norm(points, axis=1)
            mask = norms > 0.85
            points[mask] = points[mask] / norms[mask][:, np.newaxis] * 0.85
        
        # Binary search on radius
        r_low = 0.10
        r_high = 0.13
        
        # First, find a feasible radius
        mid_r = (r_low + r_high) / 2
        best_points_for_r, best_viol = simulate_annealing(
            points, mid_r, 
            start_temp=0.05, end_temp=1e-7,
            n_iterations=20000,
            time_limit=10.0
        )
        
        actual_score = compute_score(best_points_for_r)
        
        if actual_score > best_overall_score:
            best_overall_score = actual_score
            best_overall_points = best_points_for_r.copy()
        
        # Now try to improve with binary search
        for _ in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try to increase radius
            r_new = actual_score + 0.001
            if r_new > 0.13:
                break
            
            improved_points, viol = simulate_annealing(
                best_points_for_r, r_new,
                start_temp=0.02, end_temp=1e-8,
                n_iterations=15000,
                time_limit=8.0
            )
            
            improved_score = compute_score(improved_points)
            
            if improved_score > actual_score:
                actual_score = improved_score
                best_points_for_r = improved_points
                if actual_score > best_overall_score:
                    best_overall_score = actual_score
                    best_overall_points = best_points_for_r.copy()
            else:
                # Try smaller step
                r_new = actual_score + 0.0005
                if r_new <= actual_score + 1e-6:
                    break
                improved_points, viol = simulate_annealing(
                    best_points_for_r, r_new,
                    start_temp=0.01, end_temp=1e-9,
                    n_iterations=10000,
                    time_limit=6.0
                )
                improved_score = compute_score(improved_points)
                if improved_score > actual_score:
                    actual_score = improved_score
                    best_points_for_r = improved_points
                    if actual_score > best_overall_score:
                        best_overall_score = actual_score
                        best_overall_points = best_points_for_r.copy()
    
    # Final polish: run one more annealing on the best solution
    if best_overall_points is not None and time.time() - start_time < time_limit - 2:
        final_r = best_overall_score
        polished_points, _ = simulate_annealing(
            best_overall_points, final_r,
            start_temp=0.005, end_temp=1e-10,
            n_iterations=20000,
            time_limit=3.0
        )
        polished_score = compute_score(polished_points)
        if polished_score > best_overall_score:
            best_overall_points = polished_points
            best_overall_score = polished_score
    
    # Ensure validity
    if best_overall_points is None:
        # Fallback: simple spiral
        best_overall_points = np.zeros((N, 2))
        for i in range(N):
            r_init = np.sqrt(i / N) * 0.85
            theta = i * np.pi * (3 - np.sqrt(5))
            best_overall_points[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
    
    # Final validation and fix
    final_score = compute_score(best_overall_points)
    radii = np.linalg.norm(best_overall_points, axis=1)
    if np.any(radii > 1.0 - final_score):
        scale = (1.0 - final_score) / np.max(radii)
        best_overall_points *= scale
    
    return best_overall_points

# EVOLVE-BLOCK-END
