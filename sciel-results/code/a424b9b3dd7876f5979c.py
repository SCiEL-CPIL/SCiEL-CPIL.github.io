"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def construct():
    """
    Minimize logarithmic energy of N points on the unit sphere using
    Riemannian gradient descent with adaptive step size and random restarts.
    """
    
    def compute_energy(x):
        """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
        # Ensure x is on sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-15)
        
        # Pairwise squared distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Energy = sum_{i<j} log(1/dist) = -sum_{i<j} log(dist) = -0.5 * sum_{i<j} log(dist_sq)
        triu_idx = np.triu_indices(N, 1)
        energy = -0.5 * math.fsum(np.log(dist_sq[triu_idx]))
        
        return energy
    
    def compute_gradient(x):
        """
        Compute the Riemannian gradient of the energy on the sphere.
        The Euclidean gradient is grad_eucl[i] = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
        The Riemannian gradient is the projection of the Euclidean gradient onto the tangent space.
        """
        # Ensure x is on sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-15)
        
        # Pairwise differences and squared distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Euclidean gradient: grad[i] = sum_j (x_i - x_j) / dist_sq_ij
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad_eucl = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        # Project to tangent space: grad_riemann = grad_eucl - (grad_eucl · x) * x
        grad_dot_x = np.sum(grad_eucl * x, axis=1, keepdims=True)
        grad_riemann = grad_eucl - grad_dot_x * x
        
        return grad_riemann
    
    def retraction(x, grad, step_size):
        """
        Retraction from tangent space back to the sphere.
        Uses first-order approximation: x_new = normalize(x + step_size * grad)
        """
        x_new = x + step_size * grad
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x_new / norms
    
    def optimize_from_start(x0, max_iter=5000, initial_step=0.1, min_step=1e-8):
        """
        Riemannian gradient descent with adaptive step size.
        Returns (best_x, best_energy).
        """
        x = x0.copy()
        best_x = x.copy()
        best_energy = compute_energy(x)
        
        step_size = initial_step
        prev_energy = best_energy
        
        for it in range(max_iter):
            # Compute gradient
            grad = compute_gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-12:
                break
            
            # Try a step
            x_candidate = retraction(x, grad, step_size)
            energy_candidate = compute_energy(x_candidate)
            
            # Adaptive step size
            if energy_candidate < prev_energy:
                # Accept step, maybe increase step size
                x = x_candidate
                prev_energy = energy_candidate
                
                if energy_candidate < best_energy:
                    best_energy = energy_candidate
                    best_x = x.copy()
                
                # Increase step size if improvement is good
                if energy_candidate < prev_energy - 1e-10:
                    step_size = min(step_size * 1.5, 0.5)
            else:
                # Reject step, decrease step size
                step_size *= 0.5
                
                if step_size < min_step:
                    break
            
            # Periodic random kick to escape local minima
            if it % 200 == 199 and step_size > 0.01:
                rng = np.random.default_rng(it)
                kick = rng.normal(0, 0.1, x.shape)
                x = retraction(x, kick, 1.0)
                prev_energy = compute_energy(x)
        
        return best_x, best_energy
    
    # Generate initial configurations
    init_configs = []
    
    # 1. Fibonacci spiral (well-distributed starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    init_configs.append(fib)
    
    # 2. Multiple random starts
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        init_configs.append(pts)
    
    # 3. Fibonacci with varying jitter
    for jitter_scale in [0.02, 0.05, 0.1, 0.15, 0.2]:
        rng = np.random.default_rng(10000 + int(jitter_scale * 1000))
        jittered = fib + rng.normal(0, jitter_scale, fib.shape)
        jittered = jittered / np.linalg.norm(jittered, axis=1, keepdims=True)
        init_configs.append(jittered)
    
    # 4. Low-discrepancy sequences (Sobol-like via Halton)
    def halton_sequence(n, base):
        """Generate n points in [0,1) using Halton sequence."""
        seq = []
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            seq.append(r)
        return np.array(seq)
    
    # 3D Halton
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    # Map to sphere using spherical coordinates
    theta = 2 * np.pi * h1
    cos_phi = 2 * h2 - 1
    sin_phi = np.sqrt(1 - cos_phi**2)
    halton_pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])
    init_configs.append(halton_pts)
    
    # Optimize from each starting configuration
    best_energy_overall = np.inf
    best_x_overall = None
    
    start_time = time.time()
    time_budget = 110  # Leave 10 seconds margin
    
    for idx, x0 in enumerate(init_configs):
        if time.time() - start_time > time_budget:
            break
        
        x_opt, e_opt = optimize_from_start(x0, max_iter=3000, initial_step=0.05)
        
        if e_opt < best_energy_overall:
            best_energy_overall = e_opt
            best_x_overall = x_opt
    
    # Final refinement from the best solution with smaller steps
    if best_x_overall is not None and time.time() - start_time < time_budget:
        x_final, e_final = optimize_from_start(
            best_x_overall, 
            max_iter=5000, 
            initial_step=0.01, 
            min_step=1e-10
        )
        if e_final < best_energy_overall:
            best_energy_overall = e_final
            best_x_overall = x_final
    
    # Ensure output is valid
    if best_x_overall is None:
        # Fallback: use Fibonacci
        best_x_overall = fib
    
    # Final projection to ensure unit norm
    norms = np.linalg.norm(best_x_overall, axis=1, keepdims=True)
    best_x_overall = best_x_overall / np.maximum(norms, 1e-15)
    
    # Verify no points are too close
    diff = best_x_overall[:, np.newaxis, :] - best_x_overall[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Add small perturbation if points are too close
        rng = np.random.default_rng(999)
        best_x_overall = best_x_overall + rng.normal(0, 1e-6, best_x_overall.shape)
        norms = np.linalg.norm(best_x_overall, axis=1, keepdims=True)
        best_x_overall = best_x_overall / np.maximum(norms, 1e-15)
    
    return best_x_overall
# EVOLVE-BLOCK-END
