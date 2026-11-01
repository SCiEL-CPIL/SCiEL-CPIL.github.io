"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

_DEADLINE = 150.0
_SAFE_MARGIN = 10.0  # seconds


def _log_energy_and_grad(X):
    """Compute log energy sum_{i<j} log(1/|x_i-x_j|) and its gradient.
    
    X: (N, 3) array of points on unit sphere.
    Returns: (energy, grad) where grad is (N, 3) vector of negative gradient
    of energy w.r.t. X (i.e., direction to increase energy).
    """
    n = X.shape[0]
    # Pairwise distances
    # d2[i,j] = |x_i - x_j|^2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # d_ij = sqrt(d2_ij)
    # log(d_ij) = 0.5 * log(d2_ij)
    # energy = -0.5 * sum_{i!=j} 0.5 * log(d2_ij) = -0.25 * sum_{i!=j} log(d2_ij)
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    energy = -0.25 * np.sum(log_d2)
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = d/dx_i [-log(|x_i - x_j|)]
    # = -1/|x_i-x_j| * (x_i - x_j)/|x_i-x_j| = -(x_i - x_j)/|x_i-x_j|^2 = -(x_i-x_j)/d2_ij
    # For the full energy E = sum_{i<j} log(1/d_ij):
    # dE/dx_i = sum_{j != i} d/dx_i [log(1/d_ij)] (for j where i<j) + sum_{j != i} d/dx_i [log(1/d_ji)] (for j where j<i)
    # But since log(1/d_ij) = log(1/d_ji), and d/dx_i log(1/d_ij) = -(x_i - x_j)/d2_ij
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d2_ij
    #             = -n_i * sum_{j!=i} 1/d2_ij + sum_{j!=i} x_j / d2_ij
    #             = sum_{j!=i} (x_j - x_i) / d2_ij
    
    # grad[i] = sum_{j != i} (x_j - x_i) / d2_ij
    # = sum_j x_j / d2_ij - x_i * sum_j 1/d2_ij  (with j != i, d2_ii excluded)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j x_j * inv_d2[i,j] - x[i] * sum_j inv_d2[i,j]
    # First term: X^T @ inv_d2 (but inv_d2 is symmetric)
    # sum_j x_j * inv_d2[i,j] = (inv_d2 @ X)[i]
    term1 = inv_d2 @ X  # (N, 3)
    row_sums = np.sum(inv_d2, axis=1)  # (N,)
    term2 = X * row_sums[:, np.newaxis]
    
    grad = term1 - term2  # This is dE/dx_i
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_rotation_3d(rng):
    """Generate a random rotation matrix in 3D."""
    # Use QR decomposition of random Gaussian matrix
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    if d < 0:
        Q[:, 0] *= -1
    return Q


def _simulated_annealing(X, rng, time_budget=60.0):
    """Simulated annealing with random rotations."""
    t0 = time.time()
    best_X = X.copy()
    best_E, _ = _log_energy_and_grad(best_X)
    
    # Temperature schedule
    T_start = 0.1
    T_end = 1e-6
    n_steps = 50000
    
    for step in range(n_steps):
        if time.time() - t0 > time_budget:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Generate random perturbation: small random rotation
        # Scale rotation angle by temperature
        angle = np.sqrt(T) * 0.1
        R = _random_rotation_3d(rng)
        # Interpolate identity to R by angle scaling
        # Actually, let's just use a small random rotation
        # Better: generate random axis and small angle
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        theta = np.sqrt(T) * 0.05
        
        # Rotation matrix via Rodrigues
        c = np.cos(theta)
        s = np.sin(theta)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_small = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        X_new = R_small @ X
        X_new = _project_to_sphere(X_new)
        
        E_new, _ = _log_energy_and_grad(X_new)
        
        delta = E_new - best_E
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            best_X = X_new
            best_E = E_new
    
    return best_X, best_E


def _projected_gradient_descent(X, lr=0.01, max_iter=5000, tol=1e-12, time_budget=30.0):
    """Projected gradient descent on the sphere."""
    t0 = time.time()
    X = X.copy()
    E, grad = _log_energy_and_grad(X)
    
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Project gradient to tangent space: grad_t = grad - (grad . x) * x
        grad_dot = np.sum(X * grad, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot * X
        
        # Take step
        X_new = X - lr * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        E_new, grad_new = _log_energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            grad = grad_new
            # Adaptive learning rate
            if it % 100 == 0:
                lr *= 0.95
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
        
        if it > 100 and abs(E_new - E) < tol:
            break
    
    return X, E


def construct():
    t_start = time.time()
    deadline = t_start + _SAFE_MARGIN
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    X = _project_to_sphere(X)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing
    time_budget_sa = 50.0
    X, E = _simulated_annealing(X, rng, time_budget=time_budget_sa)
    
    if time.time() > deadline:
        return X
    
    # Phase 2: Projected gradient descent polish
    time_budget_pg = 30.0
    X, E = _projected_gradient_descent(X, lr=0.01, max_iter=10000, time_budget=time_budget_pg)
    
    if time.time() > deadline:
        return X
    
    # Phase 3: Additional SA from best point with lower temperature
    rng2 = np.random.default_rng(123)
    time_budget_sa2 = 20.0
    X2, E2 = _simulated_annealing(X, rng2, time_budget=time_budget_sa2)
    
    if E2 < E:
        X, E = X2, E2
    
    if time.time() > deadline:
        return X
    
    # Phase 4: Final gradient polish
    X, E = _projected_gradient_descent(X, lr=0.005, max_iter=5000, time_budget=15.0)
    
    return X
# EVOLVE-BLOCK-END
