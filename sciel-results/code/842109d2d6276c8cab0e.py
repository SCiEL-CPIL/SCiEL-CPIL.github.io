"""Improved program: velocity-Verlet dynamics + L-BFGS-B + simulated annealing + high-precision polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _log_energy(X):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    dots = X @ X.T
    np.fill_diagonal(dots, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.maximum(d2, 1e-18, out=d2)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    log_terms = -np.log(d)
    return math.fsum(log_terms.tolist())


def _energy_and_grad(X):
    """Compute log energy and gradient."""
    n = X.shape[0]
    dots = X @ X.T
    np.fill_diagonal(dots, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.maximum(d2, 1e-18, out=d2)
    dist = np.sqrt(d2)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    row_sum = inv_d2.sum(axis=1)
    weighted_sum = inv_d2 @ X
    
    grad = X * row_sum[:, None] - weighted_sum
    
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    log_terms = -np.log(d)
    energy = math.fsum(log_terms.tolist())
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n random points uniformly on sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    return _project_to_sphere(X)


def _tangent_vector(v, x):
    """Project vector v onto tangent space at point x on sphere."""
    return v - np.dot(v, x) * x


def _velocity_verlet_sphere(X, dt, n_steps, max_time=None):
    """Velocity-Verlet integration on sphere for basin exploration."""
    X = X.copy()
    V = np.zeros_like(X)
    
    for step in range(n_steps):
        if max_time is not None and time.time() > max_time:
            break
        
        E, G = _energy_and_grad(X)
        
        # Project gradient to tangent space
        G_tan = G - (G * X).sum(axis=1, keepdims=True) * X
        
        # Half-step velocity update
        V = V + 0.5 * dt * G_tan
        
        # Project velocity to tangent space
        V = V - (V * X).sum(axis=1, keepdims=True) * X
        
        # Update position
        X_new = X + dt * V
        X_new = _project_to_sphere(X_new)
        
        # Recompute gradient at new position
        E_new, G_new = _energy_and_grad(X_new)
        G_new_tan = G_new - (G_new * X_new).sum(axis=1, keepdims=True) * X_new
        
        # Second half-step velocity update
        V = V + 0.5 * dt * G_new_tan
        V = V - (V * X_new).sum(axis=1, keepdims=True) * X_new
        
        X = X_new
    
    return X


def _gradient_descent_sphere(X, lr=0.01, max_iter=500, max_time=None):
    """Gradient descent on sphere with adaptive learning rate."""
    X = X.copy()
    
    for iteration in range(max_iter):
        if max_time is not None and time.time() > max_time:
            break
        
        E, G = _energy_and_grad(X)
        grad_tangent = G - (G * X).sum(axis=1, keepdims=True) * X
        
        X_new = X - lr * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        E_new = _log_energy(X_new)
        
        if E_new < E:
            X = X_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr = lr * 0.5
        
        if lr < 1e-8:
            break
    
    return X


def _lbfgs_sphere(X, max_iter=100, max_time=None):
    """L-BFGS-B optimization with sphere constraint via re-normalization."""
    X = X.copy()
    
    def objective_and_grad(flat_x):
        X_local = _project_to_sphere(flat_x.reshape(N, 3))
        E, G = _energy_and_grad(X_local)
        # Project gradient to tangent space
        grad_tangent = G - (G * X_local).sum(axis=1, keepdims=True) * X_local
        return E, grad_tangent.flatten()
    
    result = minimize(
        objective_and_grad,
        X.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    return X_opt


def _simulated_annealing(X, T_start=0.1, T_end=0.001, n_steps=1000, max_time=None):
    """Simulated annealing with multi-point perturbations."""
    X = X.copy()
    E_current = _log_energy(X)
    
    for step in range(n_steps):
        if max_time is not None and time.time() > max_time:
            break
        
        # Linear cooling schedule
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Random perturbation: move 1-3 random points
        n_perturb = np.random.randint(1, 4)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        X_new = X.copy()
        for i in idx:
            # Random small displacement
            delta = np.random.randn(3) * T
            X_new[i] = X[i] + delta
            X_new[i] = X_new[i] / np.linalg.norm(X_new[i])
        
        E_new = _log_energy(X_new)
        
        # Accept if better or with probability exp(-(E_new - E_current)/T)
        if E_new < E_current or np.random.random() < math.exp(-(E_new - E_current) / T):
            X = X_new
            E_current = E_new
    
    return X


def _local_minimize(X, max_time=None):
    """Local minimization: gradient descent + L-BFGS-B + polish."""
    X = X.copy()
    
    # Phase 1: Gradient descent
    X = _gradient_descent_sphere(X, lr=0.01, max_iter=200, max_time=max_time)
    
    # Phase 2: L-BFGS-B
    if max_time is None or time.time() < max_time - 5:
        X = _lbfgs_sphere(X, max_iter=50, max_time=max_time)
    
    # Phase 3: Fine gradient descent polish
    X = _gradient_descent_sphere(X, lr=0.001, max_iter=100, max_time=max_time)
    
    return X


def _basin_hopping_single(X, max_time=None):
    """Single basin hopping iteration: perturb + local minimize."""
    # Perturb
    X_perturbed = X + np.random.randn(N, 3) * 0.05
    X_perturbed = _project_to_sphere(X_perturbed)
    
    # Local minimize
    X_opt = _local_minimize(X_perturbed, max_time=max_time)
    
    return X_opt


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    best_X = X.copy()
    best_E = _log_energy(best_X)
    
    np.random.seed(42)
    
    # Phase 1: Velocity-Verlet dynamics for basin exploration
    if time.time() < DEADLINE - 30:
        X_vv = _velocity_verlet_sphere(X, dt=0.01, n_steps=200, max_time=DEADLINE - 30)
        X_vv = _local_minimize(X_vv, max_time=DEADLINE - 20)
        E_vv = _log_energy(X_vv)
        if E_vv < best_E:
            best_E = E_vv
            best_X = X_vv.copy()
    
    # Phase 2: Simulated annealing
    if time.time() < DEADLINE - 20:
        X_sa = _simulated_annealing(best_X, T_start=0.1, T_end=0.001, n_steps=500, max_time=DEADLINE - 15)
        X_sa = _local_minimize(X_sa, max_time=DEADLINE - 10)
        E_sa = _log_energy(X_sa)
        if E_sa < best_E:
            best_E = E_sa
            best_X = X_sa.copy()
    
    # Phase 3: Basin hopping from multiple starts
    n_bh = 5
    for bh in range(n_bh):
        if time.time() > DEADLINE - 10:
            break
        
        # Start from best or random
        if bh % 2 == 0:
            X_start = best_X.copy()
        else:
            X_start = _random_uniform_sphere(N, seed=42 + bh)
        
        X_bh = _basin_hopping_single(X_start, max_time=DEADLINE - 5)
        E_bh = _log_energy(X_bh)
        if E_bh < best_E:
            best_E = E_bh
            best_X = X_bh.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() < DEADLINE - 2:
        X_polish = best_X.copy()
        
        # Multiple rounds of L-BFGS-B with tighter tolerances
        for round_num in range(3):
            if time.time() > DEADLINE - 1:
                break
            X_polish = _lbfgs_sphere(X_polish, max_iter=30, max_time=DEADLINE - 1)
            E_polish = _log_energy(X_polish)
            if E_polish < best_E:
                best_E = E_polish
                best_X = X_polish.copy()
        
        # Final fine gradient descent
        X_final = _gradient_descent_sphere(best_X, lr=0.0001, max_iter=50, max_time=DEADLINE - 0.5)
        E_final = _log_energy(X_final)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final.copy()
    
    return best_X
