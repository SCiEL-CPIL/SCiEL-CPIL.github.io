import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    def energy_and_grad_flat(x_flat):
        """Compute energy and gradient for flat vector (300 elements)."""
        x = x_flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-15)
        
        # Compute pairwise distances
        diffs = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        log_dists = np.log(dists)
        np.fill_diagonal(log_dists, 0.0)
        energy = -0.5 * np.sum(log_dists)
        
        # Gradient: d/dx_i [log(1/|x_i - x_j|)] = -(x_i - x_j) / |x_i - x_j|^2
        # So grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
        inv_dists_sq = 1.0 / (dists**2)
        np.fill_diagonal(inv_dists_sq, 0.0)
        weighted_diffs = diffs * inv_dists_sq[:, :, None]
        grad = np.sum(weighted_diffs, axis=1)
        
        # Project gradient to tangent space (remove radial component)
        dot_products = np.sum(grad * x, axis=1)
        grad_tan = grad - dot_products[:, None] * x
        
        return energy, grad_tan.flatten()

    def optimize_lbfgs(x_init, max_time):
        """Optimize using L-BFGS-B with time limit."""
        x_flat = x_init.flatten()
        
        def objective_and_grad(x_flat):
            if time.time() > max_time:
                raise TimeoutError()
            return energy_and_grad_flat(x_flat)
        
        try:
            result = minimize(
                objective_and_grad,
                x_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 500}
            )
            x_opt = result.x.reshape(N, 3)
            # Normalize
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            E_opt, _ = energy_and_grad_flat(x_opt.flatten())
            return x_opt, E_opt
        except (TimeoutError, Exception):
            # Return best found so far or initial
            x_opt = x_init.copy()
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            E_opt, _ = energy_and_grad_flat(x_opt.flatten())
            return x_opt, E_opt

    def fibonacci_initial():
        """Generate Fibonacci sphere initial points."""
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_initial():
        """Generate random initial points."""
        x = np.random.randn(N, 3)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def jittered_fibonacci(jitter=0.1):
        """Fibonacci sphere with random jitter."""
        x = fibonacci_initial()
        noise = np.random.randn(N, 3) * jitter
        x = x + noise
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    # Multiple restarts with different initializations
    best_overall = None
    best_E_overall = np.inf
    
    initializations = [
        fibonacci_initial(),
        random_initial(),
        jittered_fibonacci(0.05),
        jittered_fibonacci(0.1),
        jittered_fibonacci(0.2),
    ]
    
    # Allocate time per restart
    time_per_restart = (deadline - start_time - 5.0) / len(initializations)
    
    for idx, x_init in enumerate(initializations):
        if time.time() > deadline - 2.0:
            break
        
        remaining_time = deadline - time.time() - 2.0
        if remaining_time < 1.0:
            break
        
        x_opt, E_opt = optimize_lbfgs(x_init, start_time + time.time() - start_time + min(time_per_restart, remaining_time))
        
        if E_opt < best_E_overall:
            best_E_overall = E_opt
            best_overall = x_opt
    
    # Final high-precision polish on best result
    if best_overall is not None and time.time() < deadline - 3.0:
        x_final = best_overall.copy()
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            x_flat = x_final.flatten()
            E, grad = energy_and_grad_flat(x_flat)
            # Simple gradient step with small step size
            x_new = x_flat - 1e-4 * grad
            x_new = x_new.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            E_new, _ = energy_and_grad_flat(x_new.flatten())
            if E_new < E:
                x_final = x_new
                best_E_overall = E_new
                best_overall = x_final
            else:
                break
    
    return best_overall
