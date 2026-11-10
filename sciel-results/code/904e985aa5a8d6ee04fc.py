"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.90  # safety margin

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _energy_and_grad(pts):
    """Compute log energy and gradient on the sphere (tangent-space projected)."""
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} -log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # inv_d2[i,j] = 1/dists[i,j]^2 for i != j
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-14
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j diff[i,j] * inv_d2[i,j]
    grads = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    for i in range(n):
        g = grads[i]
        p = pts[i]
        grads[i] = g - np.dot(g, p) * p
    
    return energy, grads

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _local_optimize(pts, time_budget, maxiter=300):
    """Local optimization using L-BFGS-B with gradient on the sphere."""
    best_pts = _normalize(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    def objective_and_grad(x):
        p = _normalize(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    def objective(x):
        e, _ = objective_and_grad(x)
        return e
    
    def grad(x):
        _, g = objective_and_grad(x)
        return g
    
    try:
        result = minimize(
            objective, best_pts.flatten(), method='L-BFGS-B', jac=grad,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy

def _sgd_optimize(pts, time_budget, rng):
    """
    Stochastic Gradient Descent on the sphere.
    Iterates through pairs (i, j) in random order, computing the exact gradient
    contribution of that pair to points i and j, and taking a step.
    This is O(N) per step (if we just pick one pair) or O(N^2) per sweep.
    We do a sweep: pick a random permutation of all pairs? No, that's O(N^2) pairs.
    Better: pick a random point i, then a random j != i. Update i and j.
    Repeat until time budget.
    
    Energy E = sum_{i<j} -log|xi-xj|
    Grad contribution of pair (i,j) to xi: (xi - xj) / |xi - xj|^2
    Grad contribution of pair (i,j) to xj: (xj - xi) / |xi - xj|^2
    
    Step: xi = xi - lr * grad_i, then normalize.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Adaptive learning rate
    lr = 0.01
    
    while time.time() < time_budget:
        # Pick a random pair
        i = rng.integers(n)
        j = rng.integers(n)
        while j == i:
            j = rng.integers(n)
        
        xi = pts[i]
        xj = pts[j]
        
        diff = xi - xj
        d2 = np.dot(diff, diff)
        if d2 < 1e-12:
            # Too close, push apart
            rand_vec = rng.standard_normal(3)
            tangent_i = rand_vec - np.dot(rand_vec, xi) * xi
            tangent_i /= np.linalg.norm(tangent_i)
            pts[i] = _normalize(xi + 0.1 * tangent_i)
            continue
            
        d = np.sqrt(d2)
        # Gradient for xi from this pair
        grad_i = diff / d2
        # Project to tangent space at xi
        grad_i = grad_i - np.dot(grad_i, xi) * xi
        
        # Gradient for xj from this pair
        grad_j = -grad_i # (xj - xi)/d2
        # Project to tangent space at xj
        grad_j = grad_j - np.dot(grad_j, xj) * xj
        
        # Update
        pts[i] = _normalize(xi - lr * grad_i)
        pts[j] = _normalize(xj - lr * grad_j)
        
        # Decay learning rate slightly
        lr *= 0.99999

    return pts

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(12345)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Phase 1: SGD Global Search
    # Use a significant portion of the time for SGD to explore the landscape
    sgd_budget = start_time + (deadline - start_time) * 0.6
    
    # Multiple restarts of SGD from diverse starts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() >= sgd_budget:
            break
        
        if r == 0:
            init_pts = _fibonacci_spiral(N)
        else:
            init_pts = _random_points(N, rng)
            # Mix in some structure?
            if r % 2 == 0:
                 init_pts = 0.5 * init_pts + 0.5 * _fibonacci_spiral(N)
                 init_pts = _normalize(init_pts)
        
        sgd_pts = _sgd_optimize(init_pts, sgd_budget, rng)
        
        # Quick local polish to evaluate
        polished_pts, polished_energy = _local_optimize(sgd_pts, deadline, maxiter=50)
        
        if polished_energy < best_energy:
            best_pts = polished_pts
            best_energy = polished_energy
            
    # Phase 2: High-precision L-BFGS Polish
    # Use remaining time for a very careful local optimization
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Do a few more SGD steps if we have time? No, just polish.
        # Or maybe one more diverse SGD start if we have lots of time?
        if remaining_time > 30:
            # Try one more diverse start with SGD
            init_pts = _random_points(N, rng)
            sgd_pts = _sgd_optimize(init_pts, deadline - 20, rng)
            polished_pts, polished_energy = _local_optimize(sgd_pts, deadline, maxiter=50)
            if polished_energy < best_energy:
                best_pts = polished_pts
                best_energy = polished_energy
        
        # Final deep polish
        final_pts, final_energy = _local_optimize(best_pts, deadline, maxiter=1000)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy

    return best_pts
# EVOLVE-BLOCK-END
