"""Rewritten program: molecular dynamics with velocity Verlet + gradient polish for log-energy minimization."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (Euclidean): grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy using math.fsum for precision
    iu = np.triu_indices(n, k=1)
    log_inv_dist = -np.log(dist[iu])
    energy = math.fsum(log_inv_dist)
    
    # Gradient: grad_i = sum_{j!=i} (X_i - X_j) / dist_ij^2
    A = 1.0 / dist_sq
    np.fill_diagonal(A, 0.0)
    S = A.sum(axis=1)
    B = A @ X
    grad = S[:, np.newaxis] * X - B
    
    return energy, grad


def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _md_simulate(X0, steps=5000, dt=0.01, T0=0.5, T_final=0.001, cooling_rate=0.9997):
    """Molecular dynamics with velocity Verlet and adaptive cooling.
    
    The force on particle i is F_i = -grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    We add a random thermal kick each step to help escape local minima.
    """
    X = X0.copy()
    V = np.zeros_like(X)  # velocities
    n = len(X)
    
    best_energy = _energy_and_grad(X)[0]
    best_X = X.copy()
    
    T = T0
    for step in range(steps):
        # Compute forces (negative gradient)
        _, grad = _energy_and_grad(X)
        F = -grad
        
        # Velocity Verlet: half step
        V += 0.5 * dt * F
        
        # Update positions
        X += dt * V
        
        # Project to sphere (constrained dynamics)
        X = _project_to_sphere(X)
        
        # Recompute forces at new positions
        _, grad_new = _energy_and_grad(X)
        F_new = -grad_new
        
        # Remove radial component of velocity (tangent space)
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X_unit = X / X_norm
        V_radial = (V * X_unit).sum(axis=1, keepdims=True) * X_unit
        V = V - V_radial
        
        # Second half step
        V += 0.5 * dt * F_new
        
        # Add thermal noise
        V += np.random.standard_normal((n, 3)) * math.sqrt(T * dt)
        
        # Damping
        V *= 0.995
        
        # Update temperature
        T = max(T_final, T * cooling_rate)
        
        # Track best
        E = _energy_and_grad(X)[0]
        if E < best_energy:
            best_energy = E
            best_X = X.copy()
    
    return best_X, best_energy


def _gradient_descent_polish(X0, max_iter=10000, lr_init=0.05):
    """Gradient descent with adaptive learning rate and momentum for final polish."""
    X = X0.copy()
    n = len(X)
    V = np.zeros_like(X)  # momentum
    lr = lr_init
    beta = 0.9  # momentum coefficient
    best_energy = _energy_and_grad(X)[0]
    best_X = X.copy()
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(X)
        
        # Momentum update
        V = beta * V - lr * grad
        
        # Tangent space projection of velocity
        X_unit = X / np.linalg.norm(X, axis=1, keepdims=True)
        V_radial = (V * X_unit).sum(axis=1, keepdims=True) * X_unit
        V = V - V_radial
        
        X_new = X + V
        X_new = _project_to_sphere(X_new)
        
        new_energy = _energy_and_grad(X_new)[0]
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_X = X_new.copy()
        
        # Adaptive learning rate
        if new_energy < energy:
            lr = min(lr * 1.02, 0.2)
        else:
            lr = max(lr * 0.9, 1e-8)
            V *= 0.5  # reduce momentum on failure
        
        X = X_new
        
        # Convergence
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return best_X, best_energy


def _lbfgs_polish(X0, max_iter=5000):
    """Use L-BFGS on the sphere with re-projection."""
    def fun_and_grad(x_flat):
        X = x_flat.reshape(-1, 3)
        X = _project_to_sphere(X)
        E, G = _energy_and_grad(X)
        return E, G
    
    x0 = X0.flatten()
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
    X_opt = _project_to_sphere(res.x.reshape(-1, 3))
    E_opt = _energy_and_grad(X_opt)[0]
    return X_opt, E_opt


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Start from Fibonacci spiral
    X0 = _fibonacci_sphere(N)
    best_X = X0.copy()
    best_energy = _energy_and_grad(X0)[0]
    
    # Phase 1: MD simulation from Fibonacci start
    X_md, E_md = _md_simulate(X0, steps=3000, dt=0.01, T0=0.3, T_final=0.001, cooling_rate=0.9998)
    if E_md < best_energy:
        best_energy = E_md
        best_X = X_md.copy()
    
    if time.time() > deadline - 40:
        # Quick polish and return
        X_pol, E_pol = _gradient_descent_polish(best_X, max_iter=2000, lr_init=0.01)
        if E_pol < best_energy:
            best_X = X_pol
        return _project_to_sphere(best_X)
    
    # Phase 2: MD from random restarts
    rng = np.random.default_rng(123)
    for restart in range(3):
        if time.time() > deadline - 30:
            break
        
        X_rand = rng.standard_normal((N, 3))
        X_rand = _project_to_sphere(X_rand)
        
        X_md, E_md = _md_simulate(X_rand, steps=2000, dt=0.01, T0=0.5, T_final=0.001, cooling_rate=0.9997)
        if E_md < best_energy:
            best_energy = E_md
            best_X = X_md.copy()
    
    # Phase 3: Gradient descent polish on best configuration
    if time.time() < deadline - 20:
        X_pol, E_pol = _gradient_descent_polish(best_X, max_iter=5000, lr_init=0.02)
        if E_pol < best_energy:
            best_energy = E_pol
            best_X = X_pol.copy()
    
    # Phase 4: L-BFGS polish
    if time.time() < deadline - 10:
        X_lb, E_lb = _lbfgs_polish(best_X, max_iter=2000)
        if E_lb < best_energy:
            best_energy = E_lb
            best_X = X_lb.copy()
    
    # Final gradient descent with very tight tolerance
    if time.time() < deadline - 5:
        X_final, E_final = _gradient_descent_polish(best_X, max_iter=3000, lr_init=0.005)
        if E_final < best_energy:
            best_X = X_final
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    if not np.all(np.isfinite(best_X)) or np.any(np.linalg.norm(best_X, axis=1) < 1e-15):
        return _fibonacci_sphere(N)
    
    return best_X
# EVOLVE-BLOCK-END
