# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_norm = np.max(np.linalg.norm(points, axis=1))
        if max_norm > 1e-10:
            points = points / max_norm
        return points

    def nelder_mead(points, max_iter=5000, tol=1e-8):
        """
        Nelder-Mead simplex optimization for the 32-dimensional point configuration.
        This is more robust than coordinate descent for this non-smooth objective.
        """
        # Flatten points to 1D vector
        x0 = points.flatten().copy()
        d = 32  # dimension
        
        # Initialize simplex
        simplex = np.zeros((d + 1, d))
        simplex[0] = x0
        for i in range(d):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.05
        
        # Evaluate initial simplex
        fvals = np.array([compute_score(normalize(simplex[i].reshape(16, 2))) for i in range(d + 1)])
        
        best_points = normalize(x0.reshape(16, 2))
        best_score = np.max(fvals)
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < tol:
                break
            
            x_worst = simplex[-1]
            x_best = simplex[0]
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            f_r = compute_score(normalize(x_r.reshape(16, 2)))
            
            if f_r < fvals[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_worst)
                f_e = compute_score(normalize(x_e.reshape(16, 2)))
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = x_centroid - rho * (x_worst - x_centroid)
                    f_c = compute_score(normalize(x_c.reshape(16, 2)))
                else:
                    # Inside contraction
                    x_c = x_centroid + rho * (x_worst - x_centroid)
                    f_c = compute_score(normalize(x_c.reshape(16, 2)))
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        fvals[i] = compute_score(normalize(simplex[i].reshape(16, 2)))
            
            # Track best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_score:
                best_score = fvals[current_best_idx]
                best_points = normalize(simplex[current_best_idx].reshape(16, 2))
        
        return best_points, best_score

    def local_optimize(points, max_iter=3000):
        """
        Local optimization using Nelder-Mead with occasional restarts.
        """
        best_points = normalize(points.copy())
        best_score = compute_score(best_points)
        
        # Use Nelder-Mead as primary optimizer
        nm_points, nm_score = nelder_mead(best_points, max_iter=max_iter)
        
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points.copy()
        
        return best_points, best_score

    def differential_evolution_simple(bounds, max_iter=100, pop_size=40, F=0.8, CR=0.9):
        """
        Simple implementation of differential evolution for global optimization.
        """
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, n, 2))
        population = np.array([normalize(p) for p in population])
        fitness = np.array([compute_score(p) for p in population])
        
        best_idx = np.argmax(fitness)
        best_points = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for gen in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            for i in range(pop_size):
                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                cross_idx = np.random.randint(n * 2)
                for j in range(n * 2):
                    if np.random.random() < CR or j == cross_idx:
                        trial[j // 2, j % 2] = mutant[j // 2, j % 2]
                
                trial = normalize(trial)
                trial_fitness = compute_score(trial)
                
                # Selection
                if trial_fitness >= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_points = trial.copy()
            
            # Elitism: keep the best individual
            pop_best_idx = np.argmax(fitness)
            if fitness[pop_best_idx] > best_fitness:
                best_fitness = fitness[pop_best_idx]
                best_points = population[pop_best_idx].copy()
        
        return best_points, best_fitness

    def projective_warp(points, warp_strength=0.3):
        """
        Apply a radial/projective warp to the input points to create a new configuration.
        This exploits the structure of lattice-based packings by distorting them radially.
        """
        # Center the points
        center = np.mean(points, axis=0)
        pts = points - center
        
        # Compute radii
        radii = np.linalg.norm(pts, axis=1)
        max_r = np.max(radii) if np.max(radii) > 1e-10 else 1.0
        
        # Normalize radii
        r_norm = radii / max_r
        
        # Apply radial distortion: r' = r * (1 + warp_strength * (1 - r^2))
        # This pulls outer points inward and pushes inner points outward
        new_r = r_norm * (1 + warp_strength * (1 - r_norm**2))
        
        # Scale back to original max radius
        new_r = new_r * max_r
        
        # Maintain direction
        directions = pts / np.where(radii > 1e-10, radii[:, None], 1.0)
        warped_pts = directions * new_r[:, None]
        
        # Re-center
        warped_pts = warped_pts - np.mean(warped_pts, axis=0)
        
        return warped_pts

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(config1)
    
    # 2. Two concentric circles (8 points each)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.6 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config2 = np.vstack([inner_pts, outer_pts])
    initial_configs.append(config2)
    
    # 3. Three concentric circles (5, 5, 6 points)
    c1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    c2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    c3_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    c1_pts = 0.4 * np.column_stack([np.cos(c1_angles), np.sin(c1_angles)])
    c2_pts = 0.7 * np.column_stack([np.cos(c2_angles), np.sin(c2_angles)])
    c3_pts = np.column_stack([np.cos(c3_angles), np.sin(c3_angles)])
    config3 = np.vstack([c1_pts, c2_pts, c3_pts])
    initial_configs.append(config3)
    
    # 4. Square grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config4 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    initial_configs.append(config4)
    
    # 5. Hexagonal lattice subset
    # Center + 6 neighbors + 9 more in second ring
    center = np.array([[0.0, 0.0]])
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_r = 1.0
    ring1 = hex_r * np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    # Second ring: 12 points, take 9
    hex_angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring2 = 1.732 * np.column_stack([np.cos(hex_angles2), np.sin(hex_angles2)])
    config5 = np.vstack([center, ring1, ring2[:9]])
    initial_configs.append(config5)
    
    # 6. Projectively warped square grid (key new configuration)
    config6 = projective_warp(config4, warp_strength=0.25)
    initial_configs.append(config6)
    
    # 7. Projectively warped hexagonal lattice
    config7 = projective_warp(config5, warp_strength=0.2)
    initial_configs.append(config7)
    
    # 8. Random configurations
    np.random.seed(42)
    for seed in range(4):
        np.random.seed(seed)
        rand_pts = np.random.randn(n, 2)
        rand_pts = normalize(rand_pts)
        initial_configs.append(rand_pts)
    
    # Optimize each configuration
    best_overall = None
    best_overall_score = -np.inf
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Local optimization using Nelder-Mead
        opt_points, opt_score = local_optimize(config, max_iter=2000)
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall = opt_points.copy()
        
        # If we have time, try differential evolution from a few starting points
        if i < 4 and time.time() < deadline - 60:
            de_points, de_score = differential_evolution_simple(
                bounds=[-2, 2], 
                max_iter=40, 
                pop_size=30
            )
            if de_score > best_overall_score:
                best_overall_score = de_score
                best_overall = de_points.copy()
    
    # Final local optimization from the best found
    if best_overall is not None and time.time() < deadline - 20:
        final_points, final_score = local_optimize(best_overall, max_iter=3000)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_points.copy()
    
    # Final normalization
    if best_overall is not None:
        best_overall = normalize(best_overall)
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END
