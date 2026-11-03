"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy_and_grad(X):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient with respect to X (N x 3).
    
    Energy = -sum_{i<j} log(d_ij)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute all pairwise distances
    # X is (N, 3)
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero and log(0)
    # Set diagonal to 1 to avoid log(1)=0 and division issues, but we mask it out
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Energy: sum_{i<j} -log(d_ij)
    d_pairs = dist[mask]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dist_sq = 1.0 / dist_sq
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _velocity_verlet_sphere(X, dt=0.01, max_iter=200, tol=1e-10):
    """
    Velocity-Verlet dynamics on the sphere to explore basins.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    V = np.zeros_like(X_curr)  # velocity in R^3
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    for it in range(max_iter):
        # Project gradient to tangent space
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        # Velocity Verlet: half step
        V = V - 0.5 * dt * grad_tangent
        # Project velocity to tangent space
        V_dot_X = np.sum(V * X_curr, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_curr
        
        # Full position step
        X_new = X_curr + dt * V
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        # Half step velocity update
        V = V - 0.5 * dt * grad_new
        # Project velocity again
        V_dot_X = np.sum(V * X_new, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_new
        
        # Damping to help convergence
        V *= 0.999
        
        X_curr = X_new
        energy_curr = energy_new
        grad_curr = grad_new
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def _gradient_descent_sphere(X, max_iter=500, tol=1e-10, deadline=None):
    """
    Riemannian gradient descent with backtracking line search.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    step_size = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        # Take a step
        X_new = X_curr - step_size * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        # Check if energy decreased
        if energy_new < energy_curr:
            X_curr = X_new
            energy_curr = energy_new
            grad_curr = grad_new
            step_size = min(step_size * 1.5, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def _high_precision_polish(X, deadline=None):
    """
    Final high-precision polish pass with very tight tolerance.
    """
    X_curr = X.copy()
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    step_size = 0.001
    max_iter = 2000
    tol = 1e-14
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        X_new = X_curr - step_size * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        if energy_new < energy_curr:
            X_curr = X_new
            energy_curr = energy_new
            grad_curr = grad_new
        else:
            step_size *= 0.5
            if step_size < 1e-15:
                break
        
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def _parallel_tempering(X_initials, temperatures, max_iter=200, deadline=None):
    """
    Parallel tempering (replica exchange) on the sphere.
    Multiple replicas at different temperatures explore the energy landscape.
    
    Parameters:
    - X_initials: list of (N,3) arrays, one per replica
    - temperatures: list of temperature values (higher = more exploration)
    - max_iter: number of iterations
    - deadline: time deadline
    
    Returns: list of (X, energy) tuples for each replica
    """
    n_replicas = len(X_initials)
    replicas = [X.copy() for X in X_initials]
    energies = []
    grads = []
    
    # Initialize each replica
    for i in range(n_replicas):
        e, g = _log_energy_and_grad(replicas[i])
        energies.append(e)
        grads.append(g)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Each replica takes a gradient step with temperature-dependent noise
        for i in range(n_replicas):
            T = temperatures[i]
            
            # Gradient descent step
            dot_products = np.sum(grads[i] * replicas[i], axis=1)
            grad_tangent = grads[i] - dot_products[:, np.newaxis] * replicas[i]
            
            # Step size depends on temperature: higher T allows larger steps
            step_size = 0.005 * (1.0 + T)
            
            X_new = replicas[i] - step_size * grad_tangent
            
            # Add thermal noise proportional to temperature
            if T > 0.01:
                noise_scale = 0.001 * T
                noise = noise_scale * np.random.randn(N, 3)
                X_new = X_new + noise
            
            X_new = _project_to_sphere(X_new)
            
            # Accept with Metropolis criterion if energy increases
            e_new, g_new = _log_energy_and_grad(X_new)
            
            if e_new <= energies[i]:
                replicas[i] = X_new
                energies[i] = e_new
                grads[i] = g_new
            else:
                # Metropolis acceptance: accept with probability exp(-(e_new - e)/T)
                if T > 1e-6:
                    delta_e = e_new - energies[i]
                    if delta_e / T < 50:  # Avoid overflow
                        p_accept = math.exp(-delta_e / T)
                        if np.random.random() < p_accept:
                            replicas[i] = X_new
                            energies[i] = e_new
                            grads[i] = g_new
        
        # Replica exchange: try to swap adjacent replicas
        # Accept swap if it would lower the total "free energy"
        for i in range(n_replicas - 1):
            if temperatures[i] < temperatures[i + 1]:
                # Try swapping replicas i and i+1
                # Accept with probability 1 if energies are consistent with temperatures
                # Use simple criterion: accept if the lower-temperature replica
                # has higher energy than the higher-temperature replica
                if energies[i] > energies[i + 1] and temperatures[i] < temperatures[i + 1]:
                    # This is a good swap: cold replica is in a bad state, hot replica is in a good state
                    # Swap them
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    grads[i], grads[i + 1] = grads[i + 1], grads[i]
                elif np.random.random() < 0.3:
                    # Occasionally swap regardless to maintain ergodicity
                    replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    grads[i], grads[i + 1] = grads[i + 1], grads[i]
    
    return list(zip(replicas, energies))


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _log_energy_and_grad(best_X)
    
    # Phase 1: Multi-start with gradient descent (reduced to make room for parallel tempering)
    n_starts = 12
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            X = _fibonacci_spiral(N)
        elif start_idx < 6:
            # Perturbed Fibonacci spiral
            offset = np.random.uniform(-0.5, 0.5)
            X = _fibonacci_spiral(N, seed_offset=offset)
            X += 0.01 * np.random.randn(N, 3)
            X = _project_to_sphere(X)
        else:
            # Random uniform on sphere
            X = np.random.randn(N, 3)
            X = _project_to_sphere(X)
        
        # Gradient descent
        X_curr, energy_curr = _gradient_descent_sphere(X, max_iter=400, tol=1e-10, deadline=deadline - 40.0)
        
        # Velocity-Verlet dynamics for basin exploration
        if time.time() < deadline - 35.0:
            X_vv, energy_vv = _velocity_verlet_sphere(X_curr, dt=0.005, max_iter=80, tol=1e-8)
            if energy_vv < energy_curr:
                X_curr, energy_curr = X_vv, energy_vv
                # Polish after VV
                X_curr, energy_curr = _gradient_descent_sphere(X_curr, max_iter=150, tol=1e-12, deadline=deadline - 35.0)
        
        # Update best
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_X = X_curr.copy()
    
    # Phase 2: Parallel tempering to explore the landscape
    if time.time() < deadline - 20.0:
        # Create multiple replicas from the best solution and some perturbations
        n_replicas = 5
        X_initials = []
        temperatures = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        # Replica 0: best solution (cold, fine-tuning)
        X_initials.append(best_X.copy())
        
        # Replicas 1-4: increasingly perturbed versions
        for i in range(1, n_replicas):
            X_pert = best_X.copy()
            perturbation_scale = 0.005 * (i ** 1.5)
            X_pert += perturbation_scale * np.random.randn(N, 3)
            X_pert = _project_to_sphere(X_pert)
            X_initials.append(X_pert)
        
        # Run parallel tempering
        pt_results = _parallel_tempering(
            X_initials, 
            temperatures, 
            max_iter=150, 
            deadline=deadline - 15.0
        )
        
        # Find the best result from parallel tempering
        for X_pt, e_pt in pt_results:
            if e_pt < best_energy:
                # Polish this result
                X_polished, e_polished = _gradient_descent_sphere(
                    X_pt, max_iter=300, tol=1e-12, deadline=deadline - 10.0
                )
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_X = X_polished.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        X_polished, energy_polished = _high_precision_polish(best_X, deadline=deadline - 2.0)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_X = X_polished
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    return best_X
# EVOLVE-BLOCK-END
