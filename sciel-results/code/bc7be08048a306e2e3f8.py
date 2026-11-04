"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Minimizes the logarithmic energy of N points on the unit sphere.
    Uses Riemannian Gradient Descent with Adam optimizer and multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def energy_and_grad(x):
        """
        Computes the logarithmic energy and the Riemannian gradient.
        x: (N, 3) array of points on the unit sphere.
        Returns: energy (float), grad_riem (N, 3) array.
        """
        # Compute pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
        # Compute squared distances
        dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
        
        # Avoid division by zero on the diagonal
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute distances
        dist = np.sqrt(dist_sq)
        
        # Compute Energy: sum_{i<j} log(1 / dist) = - sum_{i<j} log(dist)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = -np.sum(np.log(dist[mask]))
        
        # Compute Euclidean Gradient
        # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # diff[i, j] is x_i - x_j
        # We need sum_j diff[i, j] / dist_sq[i, j]
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[np.diag_indices_from(inv_dist_sq)] = 0.0
        
        # Weighted sum of differences
        grad_eucl = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
        
        # Project to Tangent Space (Riemannian Gradient)
        # grad_riem_i = grad_eucl_i - (grad_eucl_i . x_i) x_i
        proj = np.sum(grad_eucl * x, axis=1, keepdims=True)
        grad_riem = grad_eucl - proj * x
        
        return energy, grad_riem

    def retraction(x):
        """
        Retraction onto the unit sphere (normalization).
        """
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero if a point collapses to origin (unlikely but safe)
        norms[norms == 0] = 1.0
        return x / norms

    def adam_step(x, grad, m, v, t, lr):
        """
        Performs one Adam step in the Riemannian setting.
        """
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Retract to sphere
        x_retracted = retraction(x_new)
        
        return x_retracted, m, v

    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_x = None
    best_energy = np.inf

    # Phase 1: Global Search with multiple restarts
    # We will run several restarts. Each restart runs for a fixed number of iterations.
    max_iter_phase1 = 1000
    max_restarts = 50
    
    for r_idx in range(max_restarts):
        if time.time() > deadline - 20.0:  # Reserve 20s for Phase 2
            break
        
        # Initialize
        if r_idx == 0:
            x = x0.copy()
        else:
            # Random initialization on the sphere
            x = np.random.randn(N, 3)
            x = retraction(x)
        
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        t = 0
        
        # Learning rate schedule for Phase 1
        lr_start = 0.05
        lr_end = 0.005
        lr_decay = (lr_end / lr_start) ** (1.0 / max_iter_phase1)
        lr = lr_start
        
        for it in range(max_iter_phase1):
            if time.time() > deadline - 20.0:
                break
            
            energy, grad = energy_and_grad(x)
            
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
            
            t += 1
            x, m, v = adam_step(x, grad, m, v, t, lr)
            lr *= lr_decay
        
    # Phase 2: Local Polish on the best candidate
    if best_x is not None and time.time() < deadline - 5.0:
        x = best_x.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        t = 0
        
        # Use a smaller learning rate for polishing
        max_iter_phase2 = 5000
        lr_start = 0.005
        lr_end = 0.0001
        lr_decay = (lr_end / lr_start) ** (1.0 / max_iter_phase2)
        lr = lr_start
        
        for it in range(max_iter_phase2):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = energy_and_grad(x)
            
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
            
            t += 1
            x, m, v = adam_step(x, grad, m, v, t, lr)
            lr *= lr_decay
        
        # Final check and update
        final_energy, _ = energy_and_grad(best_x)
        if final_energy < best_energy:
            best_energy = final_energy

    # If no optimization was run (should not happen), return x0
    if best_x is None:
        best_x = x0

    return best_x

# EVOLVE-BLOCK-END
