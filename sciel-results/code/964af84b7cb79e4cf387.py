# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a penalty-based energy minimization approach with binary search on radius,
    followed by direct radius maximization refinement.
    """
    N = 50
    
    def energy_and_grad(c, r):
        """
        Compute the penalty energy and its gradient.
        Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        n = N
        energy = 0.0
        grad = np.zeros_like(c)
        
        # Pairwise interactions
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Energy contribution: sum(overlap^2) / 2
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy
        
        # Gradient calculation
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        # weights[i,j] = -2 * overlap[i,j] / safe_dists[i,j]
        weights = -2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        # Boundary interactions
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # E_bound = sum (bound_violation)^2
        energy += np.sum(bound_violation ** 2)
        
        # Gradient
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        grad += grad_bound
        
        return energy, grad

    def solve_for_radius(r, c_init, maxiter=500):
        """
        Minimize energy for a fixed radius r starting from c_init.
        Returns (c_opt, min_energy).
        """
        c_flat = c_init.flatten()
        
        def obj(x):
            e, _ = energy_and_grad(x, r)
            return e
            
        def jac(x):
            _, g = energy_and_grad(x, r)
            return g.flatten()
            
        try:
            res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_init, np.inf

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice with various scales
        for scale in [0.11, 0.12, 0.13, 0.14, 0.15]:
            points = []
            for j in range(-12, 13):
                for i in range(-12, 13):
                    x = i * scale + 0.5 * scale * (j % 2)
                    y = j * scale * np.sqrt(3) / 2
                    if np.sqrt(x * x + y * y) < 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                # Sort by distance from center and take closest N
                dists = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists)[:N]
                configs.append(points[indices])
            else:
                # Fill remaining with random
                while len(points) < N:
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, 0.9)
                    points.append([radius * np.cos(angle), radius * np.sin(angle)])
                configs.append(np.array(points))
        
        # 2. Concentric rings - multiple configurations
        # Configuration A: 1, 6, 12, 18, 13
        rings_a = [1, 6, 12, 18, 13]
        radii_a = [0.0, 0.22, 0.44, 0.66, 0.85]
        points_a = []
        for k, (n_pts, rad) in enumerate(zip(rings_a, radii_a)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points_a.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points_a[:N]))
        
        # Configuration B: 1, 6, 12, 18, 13 with different radii
        radii_b = [0.0, 0.20, 0.40, 0.60, 0.80]
        points_b = []
        for k, (n_pts, rad) in enumerate(zip(rings_a, radii_b)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points_b.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points_b[:N]))
        
        # Configuration C: 1, 6, 14, 18, 11
        rings_c = [1, 6, 14, 18, 11]
        radii_c = [0.0, 0.22, 0.44, 0.66, 0.85]
        points_c = []
        for k, (n_pts, rad) in enumerate(zip(rings_c, radii_c)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points_c.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points_c[:N]))
        
        # 3. Fibonacci spiral
        points_fib = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(N):
            r = np.sqrt(i / N) * 0.95
            theta = i * golden_angle
            points_fib.append([r * np.cos(theta), r * np.sin(theta)])
        configs.append(np.array(points_fib))
        
        # 4. Random restarts
        for _ in range(8):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        return configs

    def compute_radius(c):
        """
        Compute the actual radius for a given configuration.
        """
        c = c.reshape(N, 2)
        # Minimum pairwise distance / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Minimum distance to boundary
        radii = np.linalg.norm(c, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pair_dist / 2.0, min_boundary)

    def refine_radius(c_init, r_init, max_iter=30):
        """
        Directly maximize radius by alternating between position optimization 
        and radius adjustment.
        """
        c = c_init.copy()
        r = r_init
        
        for it in range(max_iter):
            # Optimize positions for current radius
            c_opt, e_opt = solve_for_radius(r, c, maxiter=200)
            c = c_opt
            
            # Compute actual achievable radius
            r_actual = compute_radius(c)
            
            if r_actual <= r:
                # Can't increase radius, try small increase
                r_new = r + 0.0001
                c_test, e_test = solve_for_radius(r_new, c, maxiter=100)
                r_test = compute_radius(c_test)
                if r_test > r:
                    c = c_test
                    r = r_test
                else:
                    break
            else:
                # Can increase radius
                r_new = min(r_actual * 1.01, r + 0.001)
                c_test, e_test = solve_for_radius(r_new, c, maxiter=100)
                r_test = compute_radius(c_test)
                if r_test > r:
                    c = c_test
                    r = r_test
                else:
                    # Try smaller increase
                    r_new = r + 0.00005
                    c_test, e_test = solve_for_radius(r_new, c, maxiter=100)
                    r_test = compute_radius(c_test)
                    if r_test > r:
                        c = c_test
                        r = r_test
                    else:
                        break
        
        return c, r

    # Binary search for the best radius
    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.135  # Slightly above known best
    
    best_centers = None
    best_r = r_low
    
    # Initial guess for binary search
    configs = generate_initial_configurations()
    
    # Binary search with fewer iterations for speed, then refine
    for _ in range(15):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r_mid, c_init, maxiter=300)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
            
            if e_opt < 1e-8:
                feasible = True
                if e_opt < best_e_for_r:
                    best_e_for_r = e_opt
                    best_c_for_r = c_opt
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
    
    # Final polish at best_r with more iterations
    if best_centers is not None:
        c_final, e_final = solve_for_radius(best_r, best_centers, maxiter=1000)
        if e_final < 1e-8:
            best_centers = c_final
        else:
            # If final polish failed, try slightly smaller radius
            r_retry = best_r - 0.0005
            c_retry, e_retry = solve_for_radius(r_retry, best_centers, maxiter=500)
            if e_retry < 1e-8:
                best_centers = c_retry
                best_r = r_retry
    
    if best_centers is None:
        # Fallback
        best_centers = configs[0]
        best_r = compute_radius(best_centers)
    
    # Refine using direct radius maximization
    best_centers, best_r = refine_radius(best_centers, best_r, max_iter=20)
    
    # Final verification and slight adjustment if needed
    final_r = compute_radius(best_centers)
    if final_r < best_r - 1e-6:
        # Slightly reduce radius to ensure validity
        best_r = final_r
        # Try to optimize at this radius
        c_adjusted, e_adjusted = solve_for_radius(best_r, best_centers, maxiter=200)
        if compute_radius(c_adjusted) >= best_r - 1e-8:
            best_centers = c_adjusted
    
    return best_centers
# EVOLVE-BLOCK-END
