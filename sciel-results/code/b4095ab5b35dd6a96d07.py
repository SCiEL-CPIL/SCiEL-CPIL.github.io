"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.special import lpmv, lpmn

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    E = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    
    Gradient with respect to x_k:
    grad_k = sum_{j != k} (x_k - x_j) / dist_kj^3
    """
    x_sq = np.sum(points * points, axis=1)
    dist_sq = x_sq[:, np.newaxis] + x_sq[np.newaxis, :] - 2.0 * (points @ points.T)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    energy = -np.sum(np.log(d_ij))
    
    C = 1.0 / (dist_sq * dist)
    np.fill_diagonal(C, 0.0)
    
    row_sums = np.sum(C, axis=1)
    grad = row_sums[:, np.newaxis] * points - (C @ points)
    
    return energy, grad

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _optimize_spectral(points_init, time_limit):
    """
    Spectral optimization strategy:
    1. Represent points as a discrete measure.
    2. Project onto spherical harmonic basis (low order).
    3. Optimize coefficients in spectral space? 
       Actually, direct spectral optimization of point positions is non-trivial 
       because the mapping from coefficients to points is not bijective for fixed N.
       
       Alternative Interpretation of "Spectral Strategy":
       Use the spectral properties of the Laplacian to define a smoother 
       energy landscape or a better initial guess.
       
       However, the directive asks for a "different design family".
       Let's try a different optimization method that leverages structure:
       **Riemannian Trust Region Method** or **L-BFGS-B on the Stiefel Manifold**.
       
       Wait, the inspiration mentions "Fast Spherical Harmonic Transform".
       Let's implement a hybrid: 
       1. Start with Fibonacci.
       2. Use a "Spectral Smoothing" step: 
          - Compute the potential at each point using the Green's function expansion.
          - Use the gradient of the potential to guide a "flow" that maximizes repulsion 
            in a way that is spectrally stable.
          
       Actually, a more robust "different strategy" that fits the "spectral" hint 
       and is implementable within time limits is:
       **Optimization in the space of spherical harmonic coefficients of the density**, 
       followed by **matching** to points via optimal transport or simply 
       **re-initializing points based on the high-density regions of the spectral model**.
       
       But for N=100, point-based optimization is usually superior.
       
       Let's pivot to a strategy that is definitely different: 
       **Simulated Annealing with a Spectral-based Cooling Schedule** 
       or 
       **Particle Swarm Optimization (PSO) on the Sphere**.
       
       Let's go with **PSO (Particle Swarm Optimization)**. It is a global search method, 
       structurally different from gradient descent, and can escape local minima.
       We will constrain the swarm to the sphere.
    """
    start_time = time.time()
    
    # Initialize swarm
    swarm_size = 20
    points_swarm = np.zeros((swarm_size, N, 3))
    velocities = np.zeros((swarm_size, N, 3))
    
    for i in range(swarm_size):
        if i == 0:
            points_swarm[i] = points_init
        else:
            points_swarm[i] = _random_sphere(N, seed=i)
            
    # Evaluate initial energies
    energies = np.zeros(swarm_size)
    for i in range(swarm_size):
        energies[i], _ = _energy_and_grad(points_swarm[i])
        
    best_indices = np.argmin(energies)
    best_points = points_swarm[best_indices].copy()
    best_energy = energies[best_indices]
    
    p_best_points = points_swarm.copy()
    p_best_energies = energies.copy()
    
    # PSO Parameters
    w = 0.8  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    iter_count = 0
    while time.time() < time_limit - 10.0:
        iter_count += 1
        for i in range(swarm_size):
            # Update velocity
            r1 = np.random.rand(N, 3)
            r2 = np.random.rand(N, 3)
            
            # v = w*v + c1*r1*(pbest - p) + c2*r2*(gbest - p)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (p_best_points[i] - points_swarm[i]) + 
                             c2 * r2 * (best_points - points_swarm[i]))
            
            # Update position
            new_points = points_swarm[i] + 0.1 * velocities[i] # Small step factor
            
            # Project to sphere
            norms = np.linalg.norm(new_points, axis=1)
            norms[norms < 1e-9] = 1.0
            new_points = new_points / norms[:, np.newaxis]
            
            # Evaluate
            e, _ = _energy_and_grad(new_points)
            
            if e < p_best_energies[i]:
                p_best_points[i] = new_points
                p_best_energies[i] = e
                
            if e < best_energy:
                best_points = new_points
                best_energy = e
                
            points_swarm[i] = new_points
            
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_points = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_points)
    
    # Multi-start with PSO
    starts = [best_points]
    for i in range(3):
        starts.append(_random_sphere(N, seed=i+100))
        
    for start in starts:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
            
        try:
            optimized, energy = _optimize_spectral(start, deadline)
            
            # Final validation and normalization
            norms = np.linalg.norm(optimized, axis=1)
            optimized = optimized / norms[:, np.newaxis]
            
            final_energy, _ = _energy_and_grad(optimized)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = optimized.copy()
                
        except Exception:
            continue
            
    # Final Polish with Gradient Descent on the best found so far
    current = best_points
    current_energy, _ = _energy_and_grad(current)
    
    step_size = 0.01
    while time.time() < deadline - 2.0:
        energy, grad = _energy_and_grad(current)
        
        dot = np.sum(grad * current, axis=1)
        grad_tangent = grad - dot[:, np.newaxis] * current
        
        if np.linalg.norm(grad_tangent) < 1e-9:
            break
            
        alpha = step_size
        for _ in range(20):
            new_points = current + alpha * grad_tangent
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, np.newaxis]
            
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < current_energy - 1e-4 * alpha * np.sum(grad_tangent * grad_tangent):
                current = new_points
                current_energy = new_energy
                break
            alpha *= 0.5
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
                
    if current_energy < best_energy:
        best_energy = current_energy
        best_points = current

    return best_points
# EVOLVE-BLOCK-END
