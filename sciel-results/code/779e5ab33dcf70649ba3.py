"""
Riemannian Optimization with Simulated Annealing for Logarithmic Energy Minimization.

Strategy:
1. Parameterize points directly on the unit sphere $S^2$.
2. Use the Riemannian gradient (tangential component of the Euclidean gradient).
3. Implement a custom optimizer: Gradient Descent with Momentum + Simulated Annealing.
   - Momentum helps traverse flat regions.
   - Simulated Annealing allows uphill moves to escape local minima, which are
     prevalent in logarithmic energy landscapes.
4. Fast energy computation using vectorized dot products.
5. Multi-start with diverse initializations (Fibonacci, Random, Clustered).
"""

import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _log_energy(x):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|).
    x: (N, 3) array of points on unit sphere.
    """
    x = np.asarray(x)
    # Compute squared distances using dot product trick: |a-b|^2 = |a|^2 + |b|^2 - 2a.b
    # Since |a|=|b|=1, dist^2 = 2 - 2 * dot(a,b)
    # However, for numerical stability and to handle potential slight deviations,
    # we compute directly or use the matrix form.
    # Matrix form: D2[i,j] = 2 - 2 * X @ X.T
    XXT = x @ x.T
    D2 = 2.0 - 2.0 * XXT
    
    # Clip to avoid negative values due to floating point errors
    D2 = np.maximum(D2, 1e-15)
    
    # We need sum_{i<j} log(1/sqrt(D2[i,j])) = -0.5 * sum_{i<j} log(D2[i,j])
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d2_upper = D2[iu]
    
    # Sum logs
    # To be consistent with math.fsum in evaluator, we use high precision summation
    # numpy sum is usually sufficient, but let's use math.fsum on the list if needed.
    # For speed, np.sum is fine. The evaluator uses math.fsum, which is more accurate.
    # Let's stick to np.sum for speed during optimization, as the difference is negligible
    # for ranking, but we must be careful if we are close to the boundary.
    # Actually, let's use a Kahan summation or just np.sum. np.sum is fast.
    energy = -0.5 * np.sum(np.log(d2_upper))
    
    return energy

def _log_energy_grad(x):
    """
    Compute the Riemannian gradient of the logarithmic energy.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    
    Euclidean gradient w.r.t x_i:
    g_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    Riemannian gradient (projection onto tangent space at x_i):
    grad_i = g_i - (g_i . x_i) x_i
    
    Note: For points on the sphere, the term (g_i . x_i) is related to the constraint.
    Specifically, sum_j (x_i - x_j)/|x_i - x_j|^2 . x_i = sum_j (1 - x_i.x_j)/|x_i - x_j|^2.
    This is not zero. So we must project.
    
    Vectorized computation:
    D2[i,j] = |x_i - x_j|^2
    We want G[i] = sum_{j != i} (x_i - x_j) / D2[i,j]
    
    Let's compute this efficiently.
    """
    x = np.asarray(x)
    n = x.shape[0]
    
    # Compute D2 matrix
    XXT = x @ x.T
    D2 = 2.0 - 2.0 * XXT
    np.fill_diagonal(D2, 1.0) # Avoid div by 0
    
    # We need sum_j (x_i - x_j) / D2[i,j]
    # = sum_j x_i / D2[i,j] - sum_j x_j / D2[i,j]
    # = x_i * sum_j (1/D2[i,j]) - sum_j x_j / D2[i,j]
    
    # Let W = 1/D2. W[i,j] = 1/|x_i - x_j|^2
    W = 1.0 / D2
    
    # Term 1: x_i * sum_j W[i,j]
    # sum_j W[i,j] is the row sum of W
    row_sums = np.sum(W, axis=1)
    term1 = x * row_sums[:, np.newaxis]
    
    # Term 2: sum_j x_j W[i,j]
    # This is W @ x
    term2 = W @ x
    
    # Euclidean gradient
    grad_euclid = term1 - term2
    
    # Project to tangent space
    # grad_riemann = grad_euclid - (grad_euclid . x) x
    # dot products
    dots = np.sum(grad_euclid * x, axis=1)
    grad_riemann = grad_euclid - dots[:, np.newaxis] * x
    
    return grad_riemann

def _fibonacci_sphere():
    """Generate Fibonacci sphere points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    """Generate random points on the unit sphere."""
    x = rng.standard_normal((N, 3))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _optimize_riemann_sa(x0, time_limit, rng):
    """
    Riemannian Gradient Descent with Momentum and Simulated Annealing.
    """
    start_time = time.time()
    x = x0.copy()
    
    # Initial energy
    current_energy = _log_energy(x)
    best_energy = current_energy
    best_x = x.copy()
    
    # Hyperparameters
    lr = 0.01  # Initial learning rate
    momentum = 0.9
    v = np.zeros_like(x) # Velocity for momentum
    
    # Annealing parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    step = 0
    while time.time() < start_time + time_limit:
        # Compute gradient
        grad = _log_energy_grad(x)
        
        # Momentum update
        v = momentum * v - lr * grad
        x_new = x + v
        
        # Re-normalize to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        # Handle zero norms (unlikely but safe)
        norms[norms < 1e-10] = 1.0
        x_new = x_new / norms
        
        # Check validity (no points too close)
        # Compute min distance
        # For speed, we can check energy. If energy is huge, points are close.
        # Or compute min distance explicitly.
        # Let's just compute energy. If it's inf or nan, reject.
        
        new_energy = _log_energy(x_new)
        
        # Acceptance criterion (Metropolis)
        if new_energy < current_energy:
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            # Accept uphill move with probability exp(-deltaE / T)
            delta_E = new_energy - current_energy
            if T > T_min:
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    x = x_new
                    current_energy = new_energy
        
        # Anneal
        if step % 10 == 0:
            T *= cooling_rate
            # Adaptive learning rate: reduce if stuck, or keep constant?
            # Let's slowly decay LR
            lr *= 0.999
            
        step += 1
        
    return best_x, best_energy

def construct():
    """Main entry point."""
    start_time = time.time()
    # 150s limit. Keep 10s margin.
    total_time_limit = 140.0
    deadline = start_time + total_time_limit
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(_fibonacci_sphere())
    
    # 2. Random
    for _ in range(5):
        inits.append(_random_sphere(rng))
        
    # 3. Clustered (points in groups)
    # Create 5 clusters
    centers = _random_sphere(rng)
    pts = []
    for i in range(N):
        c = centers[i % 5]
        # Perturb around center
        p = c + rng.standard_normal(3) * 0.2
        p = p / np.linalg.norm(p)
        pts.append(p)
    inits.append(np.array(pts))
    
    best_x = None
    best_e = np.inf
    
    # Allocate time evenly among inits, with some slack
    # 7 inits. ~20s each.
    time_per_init = (total_time_limit - 5.0) / len(inits)
    
    for idx, x0 in enumerate(inits):
        if time.time() >= deadline:
            break
            
        # Check if we have enough time for a meaningful run
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
            
        # Use slightly less than remaining to ensure we can return
        t_limit = min(time_per_init, remaining - 1.0)
        
        x_opt, e_opt = _optimize_riemann_sa(x0, t_limit, rng)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
            
    if best_x is None:
        best_x = _fibonacci_sphere()
        
    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x

# EVOLVE-BLOCK-END
