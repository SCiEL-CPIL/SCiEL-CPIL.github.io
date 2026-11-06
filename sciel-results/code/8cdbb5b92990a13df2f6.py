"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    def compute_energy_and_grad(x):
        """
        Computes the logarithmic energy and the Riemannian gradient (tangent to the sphere).
        E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
        grad_E_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0, we'll mask later
        dist = np.sqrt(dist_sq)
        
        # Gradient calculation
        # Term (x_i - x_j) / |x_i - x_j|^2
        # Note: diff[i,j] is x_i - x_j
        inv_dist_sq = 1.0 / dist_sq
        F = diff * inv_dist_sq[:, :, np.newaxis]
        
        # grad_E = - sum_j F[i,j]
        grad_E = -np.sum(F, axis=1)
        
        # Energy calculation
        # - sum_{i<j} log(d_ij)
        # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        mask = ~np.eye(N, dtype=bool)
        log_d = np.log(dist)
        log_d[~mask] = 0.0
        total_log_d = np.sum(log_d)
        energy = -0.5 * total_log_d
        
        # Project to tangent space
        dot = np.sum(grad_E * x, axis=1, keepdims=True)
        grad_tangent = grad_E - dot * x
        
        return energy, grad_tangent

    def riemannian_gradient_descent(x, max_iters=5000):
        """
        Performs Riemannian Gradient Descent with backtracking line search.
        """
        x = x.copy()
        E, g = compute_energy_and_grad(x)
        
        for k in range(max_iters):
            if time.time() > deadline:
                break
            
            # Backtracking line search
            eta = 0.1
            # Ensure g is normalized or scaled? 
            # The magnitude of g varies. 
            # Let's normalize g to unit length for stable step size control, 
            # or use a fixed eta relative to g magnitude.
            # Actually, standard GD uses eta * g. 
            # If g is large, step is large.
            # Let's try normalizing the direction.
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-10:
                break
            direction = g / g_norm
            
            # We want to minimize E.
            # Move in direction of -grad_E? 
            # grad_tangent is grad_E projected.
            # We want to move in -grad_E direction.
            # So direction = -g / ||g||
            direction = -direction
            
            # Find step size
            # Armijo condition: E_new < E - c * eta * ||g||^2 ? 
            # Since we normalized direction, the directional derivative is -||g||.
            # E_new < E - c * eta * ||g||
            c = 1e-4
            improved = False
            for _ in range(20):
                x_new = x + eta * direction
                x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
                
                E_new, _ = compute_energy_and_grad(x_new)
                
                if E_new < E - c * eta * g_norm:
                    improved = True
                    break
                eta *= 0.5
            
            if not improved:
                break
            
            x = x_new
            E = E_new
            _, g = compute_energy_and_grad(x)
            
            # Optional: adaptive eta increase if we didn't backtrack much
            # For simplicity, keep eta fixed or decay slowly
            # eta = max(eta * 1.05, 0.1) # careful
            # Let's keep eta = 0.1 always, it's robust.
            # Or decay eta slightly to converge faster at the end?
            # No, backtracking handles it.
        
        return x, E

    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    E_fib, _ = compute_energy_and_grad(x_fib)
    if E_fib < best_energy:
        best_energy = E_fib
        best_points = x_fib.copy()
    
    # Optimize Fibonacci
    if time.time() < deadline:
        x_opt, E_opt = riemannian_gradient_descent(x_fib, max_iters=2000)
        if E_opt < best_energy:
            best_energy = E_opt
            best_points = x_opt
    
    # 2. Perturbed Fibonacci
    if time.time() < deadline:
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (N, 3))
        x_pert = x_fib + noise
        x_pert = x_pert / np.linalg.norm(x_pert, axis=1, keepdims=True)
        
        x_opt, E_opt = riemannian_gradient_descent(x_pert, max_iters=2000)
        if E_opt < best_energy:
            best_energy = E_opt
            best_points = x_opt

    # 3. Random Points
    if time.time() < deadline:
        np.random.seed(123)
        x_rand = np.random.normal(0, 1, (N, 3))
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        
        x_opt, E_opt = riemannian_gradient_descent(x_rand, max_iters=2000)
        if E_opt < best_energy:
            best_energy = E_opt
            best_points = x_opt
            
    # 4. Another Perturbed Fibonacci with different seed
    if time.time() < deadline:
        np.random.seed(999)
        noise = np.random.normal(0, 0.1, (N, 3))
        x_pert2 = x_fib + noise
        x_pert2 = x_pert2 / np.linalg.norm(x_pert2, axis=1, keepdims=True)
        
        x_opt, E_opt = riemannian_gradient_descent(x_pert2, max_iters=2000)
        if E_opt < best_energy:
            best_energy = E_opt
            best_points = x_opt

    # Final polish with smaller steps if time permits
    if time.time() < deadline - 5:
         # Re-run GD with smaller max step or more iterations on the best point
         # We can just run GD again on best_points
         x_opt, E_opt = riemannian_gradient_descent(best_points, max_iters=1000)
         if E_opt < best_energy:
             best_energy = E_opt
             best_points = x_opt

    return best_points

# EVOLVE-BLOCK-END
