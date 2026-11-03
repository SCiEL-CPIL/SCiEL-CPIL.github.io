import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    def compute_radius(centers):
        """Compute the radius for a given configuration."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        center_dists = np.sqrt(np.sum(centers**2, axis=1))
        min_boundary_dist = np.min(1.0 - center_dists)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    def hexagonal_initial_guess(spacing=0.15):
        """Create a hexagonal grid of points."""
        points = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing * np.sqrt(3)
                y = j * spacing * 1.5 + (i % 2) * spacing * 0.75 * np.sqrt(3)
                if np.hypot(x, y) < 0.85:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback to random if not enough points
            return np.random.uniform(-0.8, 0.8, (N, 2))
        
        # Select N points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)[:N]
        return points[indices]
    
    def annealing_energy(x):
        """
        Energy function for simulated annealing.
        Penalizes overlaps (distances < 2r) and container violations.
        """
        x = x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Penalty for pairs too close together
        # Use a smooth penalty that grows as distance decreases
        target_dist = 0.25  # Target minimum distance (will be adjusted)
        overlap_penalty = np.sum(np.maximum(0, target_dist - dists)**2)
        
        # Boundary penalty
        center_dists = np.sqrt(np.sum(x**2, axis=1))
        boundary_penalty = np.sum(np.maximum(0, center_dists - 0.98)**2)
        
        return overlap_penalty + boundary_penalty
    
    def simulated_annealing(x_init, t_start=1.0, t_end=0.01, n_iter=5000, target_dist=0.25):
        """Run simulated annealing on the configuration."""
        x = x_init.copy()
        best_x = x.copy()
        best_energy = annealing_energy(x)
        
        current_energy = best_energy
        current_dist = target_dist
        
        for i in range(n_iter):
            if time.time() > deadline:
                break
            
            # Temperature schedule (geometric cooling)
            t = t_start * (t_end / t_start) ** (i / n_iter)
            
            # Adjust target distance based on current best radius
            # As we find better configurations, increase the target
            current_radius = compute_radius(x)
            if current_radius > 0:
                current_dist = max(0.20, min(0.26, current_radius * 2.0 + 0.01))
            
            # Generate a random move
            move_idx = np.random.randint(0, N)
            move_size = 0.05 * (t / t_start + 0.1)
            delta = np.random.normal(0, move_size, 2)
            x_new = x.copy()
            x_new[move_idx] += delta
            
            # Keep inside container
            norm = np.linalg.norm(x_new[move_idx])
            if norm > 0.98:
                x_new[move_idx] *= 0.98 / norm
            
            # Compute energy change
            e_old = annealing_energy(x)
            e_new = annealing_energy(x_new)
            delta_e = e_new - e_old
            
            # Accept or reject
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(t, 1e-10)):
                x = x_new
                current_energy = e_new
                
                # Track best
                r_new = compute_radius(x)
                r_best = compute_radius(best_x)
                if r_new > r_best:
                    best_x = x.copy()
                    best_energy = e_new
        
        return best_x
    
    # Initial guess
    x0 = hexagonal_initial_guess()
    
    best_centers = x0.copy()
    best_radius = compute_radius(x0)
    
    # Run simulated annealing with multiple restarts
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Jitter initial guess for diversity
        jitter_scale = 0.02 + restart * 0.01
        jitter = np.random.normal(0, jitter_scale, x0.shape)
        x_init = x0 + jitter
        
        # Keep inside container
        for i in range(N):
            norm = np.linalg.norm(x_init[i])
            if norm > 0.95:
                x_init[i] *= 0.95 / norm
        
        # Run annealing
        x_opt = simulated_annealing(x_init, t_start=0.5, t_end=0.01, n_iter=3000)
        
        current_radius = compute_radius(x_opt)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = x_opt
    
    # Final refinement with local optimization
    if time.time() < deadline - 20:
        def smooth_objective(x):
            x = x.reshape(-1, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, 1.0)
            
            # Soft minimum: use exponential to emphasize small distances
            alpha = 50.0
            soft_min = -np.log(np.sum(np.exp(-alpha * dists))) / alpha
            
            # Boundary constraint
            center_dists = np.sqrt(np.sum(x**2, axis=1))
            boundary_penalty = np.sum(np.maximum(0, center_dists - 0.99)**2)
            
            return soft_min + boundary_penalty
        
        try:
            res = minimize(smooth_objective, best_centers, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            x_refined = res.x.reshape(-1, 2)
            
            # Ensure validity
            center_dists = np.sqrt(np.sum(x_refined**2, axis=1))
            if np.all(center_dists <= 0.999):
                refined_radius = compute_radius(x_refined)
                if refined_radius > best_radius:
                    best_radius = refined_radius
                    best_centers = x_refined
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
