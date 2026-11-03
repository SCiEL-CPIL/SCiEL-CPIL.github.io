"""Optimize 150 points on the unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient with respect to X (N x 3).
    
    dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise differences: D[i,j] = x_i - x_j
    D = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    # Squared distances
    dist_sq = np.einsum('ijk,ijk->ij', D, D)  # (N, N)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = sum_j x_j * inv_dist_sq[i,j] - x_i * sum_j inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ X  # (N, 3)
    grad = weighted_sum - X * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _tangent_project(X, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * X, axis=1, keepdims=True)
    return grad - g_dot_n * X


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate Fibonacci sphere points with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(X, max_iters=5000, lr_init=0.01, lr_decay=0.999, tol=1e-12, time_limit=None):
    """
    Gradient descent on the sphere using projected gradient descent with momentum.
    """
    X = _project_to_sphere(X.copy())
    energy, grad = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    lr = lr_init
    
    # Momentum
    velocity = np.zeros_like(X)
    momentum = 0.9
    
    start_time = time.time()
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Project gradient to tangent space
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-15)
        grad_tangent = _tangent_project(X_norm, grad)
        
        # Check convergence
        g_norm = np.linalg.norm(grad_tangent)
        if g_norm < 1e-10 and it > 100:
            break
        
        # Momentum update
        velocity = momentum * velocity - lr * grad_tangent
        
        # Update
        X_new = X + velocity
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _energy_and_grad(X_new)
        
        if energy_new < energy:
            X = X_new
            energy = energy_new
            grad = grad_new
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
        else:
            # Backtrack
            lr *= 0.5
            velocity *= 0.5
        
        # Decay learning rate slowly
        if it % 100 == 0:
            lr = max(lr * lr_decay, 1e-8)
    
    return best_X, best_energy


def _simulated_annealing(X, max_iters=3000, T_init=0.01, T_min=1e-6, T_decay=0.999, time_limit=None, rng=None):
    """
    Simulated annealing on the sphere with batch moves.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    X = _project_to_sphere(X.copy())
    energy, _ = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    T = T_init
    
    start_time = time.time()
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Propose a random perturbation
        # Perturb a few random points
        n_perturb = int(rng.integers(1, 6))
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        X_new = X.copy()
        for i in idx:
            # Random direction in tangent space
            rand_dir = rng.standard_normal(3)
            rand_dir -= np.dot(rand_dir, X[i]) * X[i]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-15:
                rand_dir /= norm
            else:
                rand_dir = rng.standard_normal(3)
                rand_dir -= np.dot(rand_dir, X[i]) * X[i]
                norm = np.linalg.norm(rand_dir)
                if norm > 1e-15:
                    rand_dir /= norm
            
            # Move along random direction
            step_size = T * 0.3
            new_pt = X[i].copy()
            new_pt += step_size * rand_dir
            new_pt = new_pt / np.linalg.norm(new_pt)
            X_new[i] = new_pt
        
        energy_new, _ = _energy_and_grad(X_new)
        
        delta = energy_new - energy
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            X = X_new
            energy = energy_new
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
        
        T = max(T * T_decay, T_min)
    
    return best_X, best_energy


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    # Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    # Rotated Fibonacci spheres
    for angle in [0.3, 0.7, 1.1, 1.5]:
        X = _fibonacci_sphere(N)
        # Rotate by random angle
        theta = angle
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        starts.append(X @ R.T)
    # Random points
    for _ in range(4):
        X = rng.normal(size=(N, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        starts.append(X)
    
    best_X = None
    best_energy = np.inf
    
    # Phase 1: Gradient descent from each start
    for i, X0 in enumerate(starts):
        if time.time() - start_time > deadline:
            break
        time_remaining = deadline - (time.time() - start_time)
        X_opt, E_opt = _gradient_descent(X0, max_iters=3000, lr_init=0.01, time_limit=time_remaining * 0.8)
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt
    
    # Phase 2: Simulated annealing from best
    if best_X is not None and time.time() - start_time < deadline:
        time_remaining = deadline - (time.time() - start_time)
        X_opt, E_opt = _simulated_annealing(best_X, max_iters=2000, T_init=0.005, time_limit=time_remaining * 0.5, rng=rng)
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt
    
    # Phase 3: Final gradient descent polish
    if best_X is not None and time.time() - start_time < deadline:
        time_remaining = deadline - (time.time() - start_time)
        X_opt, E_opt = _gradient_descent(best_X, max_iters=5000, lr_init=0.005, lr_decay=0.9995, tol=1e-14, time_limit=time_remaining * 0.9)
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt
    
    # Final projection and validation
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    best_X = _project_to_sphere(best_X)
    
    # Ensure no points are too close
    for _ in range(10):
        D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
        dist = np.linalg.norm(D, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist = dist[iu].min()
        if min_dist > 1e-9:
            break
        # Nudge the closest pair apart
        i, j = np.unravel_index(np.argmin(dist[iu]), (N, N))
        if i > j:
            i, j = j, i
        direction = best_X[i] - best_X[j]
        norm = np.linalg.norm(direction)
        if norm > 1e-15:
            direction /= norm
        else:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
        best_X[i] += 0.01 * direction
        best_X[j] -= 0.01 * direction
        best_X = _project_to_sphere(best_X)
    
    # Final check
    D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(D, axis=2)
    iu = np.triu_indices(N, k=1)
    assert dist[iu].min() > 1e-9, "Points too close"
    assert np.all(np.isfinite(best_X)), "Non-finite values"
    assert best_X.shape == (N, 3), "Wrong shape"
    
    return best_X

# EVOLVE-BLOCK-END
