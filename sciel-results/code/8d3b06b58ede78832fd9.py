# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


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

    def soft_min_max_score(points, beta=20.0):
        """Smooth surrogate using log-sum-exp for soft min and max."""
        d = pdist(points)
        if d.size == 0:
            return 0.0
        # Soft max: log(sum(exp(beta*d))) / beta
        max_d = np.max(d)
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * (d - max_d)))) + max_d
        # Soft min: -log(sum(exp(-beta*d))) / beta
        min_d = np.min(d)
        soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * (d - min_d)))) - min_d
        if soft_max == 0:
            return 0.0
        ratio = soft_min / soft_max
        return ratio * ratio

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points

    def coordinate_ascent(points, step_sizes, max_rounds=50):
        """Coordinate ascent with random directions and decreasing step sizes."""
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        for step_size in step_sizes:
            if time.time() >= deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 5:
                        break
                    # Try moving point i in random directions
                    for _ in range(20):
                        if time.time() >= deadline - 5:
                            break
                        direction = np.random.randn(2)
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                        new_points = best_points.copy()
                        new_points[i] += direction * step_size
                        score = compute_score(new_points)
                        if score > best_score:
                            best_points = new_points
                            best_score = score
                            improved = True
                            break
        return best_points, best_score

    def gradient_ascent_smooth(points, lr=0.01, steps=500, beta=20.0):
        """Gradient ascent using smooth surrogate objective."""
        best_points = points.copy()
        best_score = compute_score(points)
        current = points.copy()
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient via central differences on smooth objective
            grad = np.zeros_like(current)
            eps = 1e-4
            for i in range(n):
                for j in range(2):
                    perturbed = current.copy()
                    perturbed[i, j] += eps
                    f_plus = soft_min_max_score(perturbed, beta)
                    
                    perturbed = current.copy()
                    perturbed[i, j] -= eps
                    f_minus = soft_min_max_score(perturbed, beta)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            
            # Update points
            current = current + lr * grad
            current = normalize(current)
            
            # Track best by exact score
            current_score = compute_score(current)
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
            
            # Decay learning rate
            lr *= 0.998
        
        return best_points, best_score

    def gradient_ascent_exact(points, lr=0.005, steps=300):
        """Gradient ascent using exact (non-smooth) objective with small steps."""
        best_points = points.copy()
        best_score = compute_score(points)
        current = points.copy()
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient via central differences on exact objective
            grad = np.zeros_like(current)
            eps = 1e-5
            for i in range(n):
                for j in range(2):
                    perturbed = current.copy()
                    perturbed[i, j] += eps
                    f_plus = compute_score(perturbed)
                    
                    perturbed = current.copy()
                    perturbed[i, j] -= eps
                    f_minus = compute_score(perturbed)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            
            # Update points
            current = current + lr * grad
            current = normalize(current)
            
            current_score = compute_score(current)
            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()
            
            lr *= 0.997
        
        return best_points, best_score

    def lbfgs_optimize(points, beta=20.0, maxiter=1000):
        """Use L-BFGS-B to optimize the smooth surrogate."""
        def objective_flat(x):
            pts = x.reshape(n, 2)
            return -soft_min_max_score(pts, beta)
        
        def gradient_flat(x):
            pts = x.reshape(n, 2)
            grad = np.zeros_like(pts)
            eps = 1e-4
            for i in range(n):
                for j in range(2):
                    p_plus = pts.copy()
                    p_plus[i, j] += eps
                    f_plus = soft_min_max_score(p_plus, beta)
                    
                    p_minus = pts.copy()
                    p_minus[i, j] -= eps
                    f_minus = soft_min_max_score(p_minus, beta)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            return grad.flatten()
        
        x0 = points.flatten()
        result = minimize(
            objective_flat, 
            x0, 
            method='L-BFGS-B', 
            jac=gradient_flat,
            options={'maxiter': maxiter, 'maxfun': 100000, 'maxls': 50}
        )
        
        if result.success or result.fun < objective_flat(x0):
            best_points = result.x.reshape(n, 2)
            best_points = normalize(best_points)
            best_score = compute_score(best_points)
            return best_points, best_score
        
        return points, compute_score(points)

    def nelder_mead_optimize(points, maxiter=2000):
        """Use Nelder-Mead to polish the exact objective."""
        def objective_flat(x):
            pts = x.reshape(n, 2)
            pts = normalize(pts)
            return -compute_score(pts)
        
        x0 = points.flatten()
        result = minimize(
            objective_flat, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'maxfev': 50000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < objective_flat(x0):
            best_points = result.x.reshape(n, 2)
            best_points = normalize(best_points)
            best_score = compute_score(best_points)
            return best_points, best_score
        
        return points, compute_score(points)

    def simulated_annealing(points, max_iter=2000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.05
        cooling_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            perturbed = current_points + np.random.randn(n, 2) * temperature
            perturbed = normalize(perturbed)
            
            new_score = compute_score(perturbed)
            
            if new_score > current_score or (temperature > 1e-8 and np.random.random() < np.exp((new_score - current_score) / (temperature + 1e-10))):
                current_points = perturbed
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = perturbed.copy()
            
            temperature *= cooling_rate
        
        return best_points, best_score

    def lattice_warp(points, warp_strength=1.0):
        """Apply a radial/projective warp to a lattice configuration."""
        center = np.mean(points, axis=0)
        pts = points - center
        r = np.linalg.norm(pts, axis=1, keepdims=True)
        max_r = np.max(r)
        if max_r == 0:
            return points
        r_norm = r / max_r
        # Radial distortion: push points outward/inward based on radius
        warp = 1.0 + warp_strength * (r_norm ** 2 - 0.5)
        warped = pts * warp
        return warped + center

    def optimize_single(config, time_budget=None):
        """Run full optimization pipeline on a single initial configuration."""
        if time_budget is None:
            time_budget = deadline - time.time() - 15
        
        # Phase 1: Coordinate ascent
        ca_points, ca_score = coordinate_ascent(config, [0.1, 0.05, 0.02, 0.01, 0.005], max_rounds=25)
        best_pts = ca_points.copy()
        best_sc = ca_score
        
        # Phase 2: L-BFGS-B with smooth surrogate
        if time.time() < deadline - 15:
            lb_points, lb_score = lbfgs_optimize(ca_points, beta=25.0, maxiter=500)
            if lb_score > best_sc:
                best_pts = lb_points.copy()
                best_sc = lb_score
        
        # Phase 3: Gradient ascent with smooth surrogate
        if time.time() < deadline - 12:
            ga_points, ga_score = gradient_ascent_smooth(best_pts, lr=0.015, steps=200, beta=25.0)
            if ga_score > best_sc:
                best_pts = ga_points.copy()
                best_sc = ga_score
        
        # Phase 4: Gradient ascent with exact objective (polish)
        if time.time() < deadline - 10:
            ge_points, ge_score = gradient_ascent_exact(best_pts, lr=0.003, steps=150)
            if ge_score > best_sc:
                best_pts = ge_points.copy()
                best_sc = ge_score
        
        # Phase 5: Nelder-Mead polish
        if time.time() < deadline - 8:
            nm_points, nm_score = nelder_mead_optimize(best_pts, maxiter=1000)
            if nm_score > best_sc:
                best_pts = nm_points.copy()
                best_sc = nm_score
        
        # Phase 6: Simulated annealing
        if time.time() < deadline - 6:
            sa_points, sa_score = simulated_annealing(best_pts, max_iter=800)
            if sa_score > best_sc:
                best_pts = sa_points.copy()
                best_sc = sa_score
        
        # Phase 7: Final coordinate ascent polish
        if time.time() < deadline - 4:
            fp_points, fp_score = coordinate_ascent(best_pts, [0.01, 0.005, 0.002, 0.001, 0.0005], max_rounds=30)
            if fp_score > best_sc:
                best_pts = fp_points.copy()
                best_sc = fp_score
        
        # Phase 8: Final exact gradient polish
        if time.time() < deadline - 2:
            final_points, final_score = gradient_ascent_exact(best_pts, lr=0.001, steps=80)
            if final_score > best_sc:
                best_pts = final_points.copy()
                best_sc = final_score
        
        # Phase 9: Final Nelder-Mead
        if time.time() < deadline - 1:
            final_nm, final_nm_score = nelder_mead_optimize(best_pts, maxiter=500)
            if final_nm_score > best_sc:
                best_pts = final_nm.copy()
                best_sc = final_nm_score
        
        return best_pts, best_sc

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(config1)
    
    # 2. Square grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config2 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    config2 = normalize(config2)
    initial_configs.append(config2)
    
    # 3. Two concentric rings (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config3 = np.vstack([inner_pts, outer_pts])
    config3 = normalize(config3)
    initial_configs.append(config3)
    
    # 4. Three concentric rings
    ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring3_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1_pts = 0.4 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    r2_pts = 0.7 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    r3_pts = 1.0 * np.column_stack([np.cos(ring3_angles), np.sin(ring3_angles)])
    config4 = np.vstack([r1_pts, r2_pts, r3_pts])
    config4 = normalize(config4)
    initial_configs.append(config4)
    
    # 5. Random with seed
    np.random.seed(42)
    config5 = np.random.randn(n, 2)
    config5 = normalize(config5)
    initial_configs.append(config5)
    
    # 6. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:16])
    config6 = normalize(hex_pts)
    initial_configs.append(config6)
    
    # 7. Lattice with radial warp (projective-like distortion)
    config7 = lattice_warp(config2, warp_strength=0.3)
    config7 = normalize(config7)
    initial_configs.append(config7)
    
    # 8. Lattice with stronger warp
    config8 = lattice_warp(config2, warp_strength=0.6)
    config8 = normalize(config8)
    initial_configs.append(config8)
    
    # 9. Two rings with different radii
    inner_angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts2 = 0.6 * np.column_stack([np.cos(inner_angles2), np.sin(inner_angles2)])
    outer_pts2 = np.column_stack([np.cos(outer_angles2), np.sin(outer_angles2)])
    config9 = np.vstack([inner_pts2, outer_pts2])
    config9 = normalize(config9)
    initial_configs.append(config9)
    
    # 10. Random with different seed
    np.random.seed(123)
    config10 = np.random.randn(n, 2)
    config10 = normalize(config10)
    initial_configs.append(config10)
    
    # Optimize each configuration
    best_overall = None
    best_overall_score = -np.inf
    
    for i, config in enumerate(initial_configs):
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        
        # Allocate time proportionally, giving more time to early configs
        # but ensuring all get a chance
        config_result, config_score = optimize_single(config)
        
        if config_score > best_overall_score:
            best_overall_score = config_score
            best_overall = config_result.copy()
    
    # Final normalization
    if best_overall is not None:
        best_overall = normalize(best_overall)
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END
