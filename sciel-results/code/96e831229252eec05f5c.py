# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a smooth surrogate objective function to directly maximize the minimum distance,
    followed by a local refinement using the standard penalty method to ensure validity.
    """
    N = 50
    
    def smooth_min_distance(c, power=20):
        """
        Computes a smooth approximation of the minimum distance to boundaries and other circles.
        d_min = min( min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|) )
        
        We approximate min(x_1, ..., x_n) using the power mean:
        M_p(x) = ( (1/n) * sum(x_i^{-p}) )^{-1/p} for p -> infinity.
        Here we use a large p to make it sharp.
        """
        c = c.reshape(N, 2)
        
        # Distances to other circles (pairwise)
        # We need d_ij / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Distances to boundary
        radii = np.linalg.norm(c, axis=1)
        dists_bound = 1.0 - radii
        
        # Combine all distances
        # We want min(d_ij/2, dists_bound)
        # Let's collect all relevant distances
        # Number of pairs: N*(N-1)/2
        # Number of boundary distances: N
        # Total terms: N*(N-1)/2 + N
        
        # To avoid 0 division, add small epsilon
        eps = 1e-12
        
        # Pairwise terms: d_ij / 2
        pair_dists = dists / 2.0
        pair_terms = 1.0 / (pair_dists + eps)
        
        # Boundary terms: 1 - |c_i|
        bound_terms = 1.0 / (dists_bound + eps)
        
        # Sum of inverse powers
        sum_inv_p = np.sum(pair_terms ** power) + np.sum(bound_terms ** power)
        
        # Power mean
        # M_p = ( (1/n_terms) * sum )^(-1/p)
        n_terms = N * (N - 1) / 2.0 + N
        min_dist_approx = ( (1.0 / n_terms) * sum_inv_p ) ** (-1.0 / power)
        
        return min_dist_approx

    def objective(x):
        """
        Objective function for minimization (we want to maximize distance).
        Returns -smooth_min_distance.
        """
        c = x.reshape(N, 2)
        return -smooth_min_distance(c)

    def grad_objective(x):
        """
        Gradient of the objective function.
        We use numerical gradient for simplicity and robustness, 
        or analytical if possible. Given the complexity of the power mean gradient,
        and that L-BFGS can handle finite differences well enough for N=50,
        we might stick to finite differences if analytical is too complex.
        However, L-BFGS is much faster with analytical gradients.
        
        Let's implement the analytical gradient for the power mean.
        f = ( (1/n) * sum (1/d_k)^p )^(-1/p)
        Let S = sum (1/d_k)^p
        f = ( S/n )^(-1/p) = n^(1/p) * S^(-1/p)
        
        df/dx_i = n^(1/p) * (-1/p) * S^(-1/p - 1) * dS/dx_i
        
        dS/dx_i = sum_k p * (1/d_k)^(p-1) * (-1) * d_k^(-2) * (dx_k/dx_i)
        where d_k is the distance corresponding to term k.
        
        This is doable but tedious. Given the 120s limit, finite difference might be risky if the landscape is noisy.
        Let's try to use the penalty gradient from the original code but applied to a fixed "target" radius that we update?
        
        Alternative: Use the original penalty energy but with a "soft" target radius that is updated during the optimization?
        
        Let's stick to a simpler robust method:
        1. Generate good initial configs.
        2. Use L-BFGS with the **original penalty energy** but for a **specific r**.
        3. Instead of binary search, we use a **hill-climbing** on r.
        4. Start with r=0.12.
        5. Try to solve. If success (energy < tol), increase r by small step.
        6. If fail, decrease r or perturb and retry.
        
        Actually, the current code does binary search which is robust. The issue is likely local minima.
        Let's enhance the current code by adding **Simulated Annealing** to the radius search.
        
        Revised Plan:
        1. Keep the `energy_and_grad` from the original.
        2. Replace the binary search loop with an annealing loop on `r`.
        3. Start `r` low.
        4. In each step, try to minimize energy for current `r`.
        5. If energy is low, increase `r` (cooling).
        6. If energy is high, add random noise to positions (heating) and retry or decrease `r`.
        """
        # This block is a placeholder for the actual implementation below
        pass

    # Re-implementing the logic with the "Annealing on Radius" strategy
    
    def energy_and_grad(c, r):
        c = c.reshape(N, 2)
        n = N
        energy = 0.0
        grad = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlap = np.maximum(0, 2 * r - dists)
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        weights = -2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        energy += np.sum(bound_violation ** 2)
        
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        grad += grad_bound
        
        return energy, grad

    def solve_for_radius(r, c_init, max_iter=200):
        c_flat = c_init.flatten()
        
        def obj(x):
            e, _ = energy_and_grad(x, r)
            return e
            
        def jac(x):
            _, g = energy_and_grad(x, r)
            return g.flatten()
            
        try:
            res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_init, np.inf

    def generate_initial_configurations():
        configs = []
        # 1. Random
        for _ in range(10):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        # 2. Hexagonal
        scale = 0.15
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            configs.append(points[indices])
            
        return configs

    np.random.seed(42)
    configs = generate_initial_configurations()
    
    best_r = 0.12
    best_centers = configs[0]
    
    # Annealing search for r
    # We will try to increase r from 0.12 to 0.126
    # Step size 0.0005
    current_r = 0.12
    max_r = 0.126
    step = 0.0005
    
    # Use the best initial config
    c_current = configs[0]
    
    # First, let's find a good starting point by minimizing energy at r=0.12
    c_current, e = solve_for_radius(current_r, c_current)
    
    # Now anneal
    T = 0.01 # Temperature (noise magnitude)
    
    while current_r < max_r:
        c_opt, e_opt = solve_for_radius(current_r, c_current)
        
        if e_opt < 1e-6:
            # Success, increase r
            current_r += step
            c_current = c_opt
        else:
            # Failure, add noise and retry or decrease r
            # Add noise
            noise = np.random.normal(0, T, (N, 2))
            c_noisy = c_current + noise
            # Project back to valid region roughly? Just let optimizer handle it
            c_retry, e_retry = solve_for_radius(current_r, c_noisy)
            
            if e_retry < 1e-6:
                current_r += step
                c_current = c_retry
            else:
                # Decrease r slightly to recover
                current_r -= step * 0.5
                if current_r < 0.12:
                    current_r = 0.12
                # Update temperature
                T *= 0.95
                
        # Update best if we found a valid config at a higher r
        # The solver returns a config that minimizes energy for current_r.
        # If energy is low, it's a valid packing for current_r.
        if e_opt < 1e-6 and current_r > best_r:
            best_r = current_r
            best_centers = c_opt
            
    # Final polish
    c_final, e_final = solve_for_radius(best_r, best_centers, max_iter=500)
    if e_final < 1e-6:
        best_centers = c_final

    return best_centers
# EVOLVE-BLOCK-END
