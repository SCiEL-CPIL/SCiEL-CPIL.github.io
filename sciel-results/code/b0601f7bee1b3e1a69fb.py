"""
Strategy: Differentiable neural network parameterization of circle centers.
Instead of directly optimizing 600 coordinates, we represent the 300 centers as
the output of a small fully-connected neural network with fixed random weights
and a learnable bias/output layer. This creates a smooth manifold of valid
configurations that can break symmetry and escape local minima. We optimize
the latent parameters with L-BFGS-B on a smooth soft-min objective.

The key insight: by constraining the solution to lie on a neural network's
output manifold, we implicitly regularize the search space, which can help
escape the many symmetric local minima that plague direct coordinate optimization
for circle packing problems.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300
DEADLINE = 118.0  # Safety margin from 150s limit


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    if best_points is not None and len(best_points) == N:
        return best_points

    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]


def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)

    return min(min_boundary, min_pair / 2.0)


def _build_nn(latent_dim=20, hidden_dim=50):
    """
    Build a simple neural network parameterization.
    Input: latent vector of size latent_dim
    Hidden layer: tanh activation
    Output layer: linear, produces N*2 coordinates
    """
    rng = np.random.default_rng(42)
    # Fixed random weights for hidden layer
    W1 = rng.standard_normal((latent_dim, hidden_dim)) * np.sqrt(2.0 / latent_dim)
    b1 = rng.standard_normal(hidden_dim) * 0.1
    # Output layer weights (will be optimized)
    W2 = rng.standard_normal((hidden_dim, N * 2)) * np.sqrt(2.0 / hidden_dim)
    b2 = rng.standard_normal(N * 2) * 0.01
    return W1, b1, W2, b2


def _nn_forward(latent, W1, b1, W2, b2):
    """Forward pass of the neural network."""
    h = np.tanh(latent @ W1 + b1)
    out = h @ W2 + b2
    return out.reshape(N, 2)


def _nn_jacobian(latent, W1, b1, W2, b2):
    """
    Compute the Jacobian of the network output w.r.t. the latent variables.
    Returns: J of shape (N*2, latent_dim)
    """
    h = np.tanh(latent @ W1 + b1)
    # tanh' = 1 - tanh^2
    h_prime = 1.0 - h * h
    # d(h)/d(latent) = diag(h_prime) @ W1
    # d(out)/d(h) = W2
    # d(out)/d(latent) = W2 @ diag(h_prime) @ W1
    # Shape: (N*2, hidden) @ (hidden, hidden) @ (hidden, latent)
    J = (W2 * h_prime[:, np.newaxis]) @ W1.T
    return J


def _optimize_nn(points_init, time_limit, latent_dim=20, hidden_dim=50):
    """
    Optimize the neural network parameters to find a good packing.
    We optimize the latent input AND the output layer weights/biases.
    """
    start_time = time.time()
    
    W1, b1, W2, b2 = _build_nn(latent_dim, hidden_dim)
    
    # Initialize latent to zero
    latent = np.zeros(latent_dim)
    
    # We'll optimize: latent, W2, b2 (W1, b1 are fixed for stability)
    # Total parameters: latent_dim + hidden_dim*N*2 + N*2
    
    # Start with a target radius slightly below known best
    r_target = 0.052
    
    def objective(params):
        """Combined objective: minimize penalty for given r_target."""
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return 1e10
        
        # Unpack parameters
        lat = params[:latent_dim]
        w2_flat = params[latent_dim:latent_dim + hidden_dim * N * 2].reshape(hidden_dim, N * 2)
        b2_flat = params[latent_dim + hidden_dim * N * 2:]
        
        # Forward pass
        h = np.tanh(lat @ W1 + b1)
        out = (h @ w2_flat + b2_flat).reshape(N, 2)
        
        # Compute penalty
        pts = out
        dists_to_origin = np.hypot(pts[:, 0], pts[:, 1])
        boundary_viol = np.maximum(0.0, dists_to_origin - (1.0 - r_target))
        boundary_pen = np.sum(boundary_viol * boundary_viol)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        cutoff = 3.0 * r_target
        mask = dists < cutoff
        pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
        pair_viol[~mask] = 0.0
        pair_pen = np.sum(pair_viol * pair_viol) / 2.0
        
        return boundary_pen + pair_pen
    
    def gradient(params):
        """Gradient of the objective w.r.t. parameters."""
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return np.zeros_like(params)
        
        # Unpack parameters
        lat = params[:latent_dim]
        w2_flat = params[latent_dim:latent_dim + hidden_dim * N * 2].reshape(hidden_dim, N * 2)
        b2_flat = params[latent_dim + hidden_dim * N * 2:]
        
        # Forward pass
        h = np.tanh(lat @ W1 + b1)
        h_prime = 1.0 - h * h
        out = (h @ w2_flat + b2_flat).reshape(N, 2)
        pts = out
        
        # Compute penalty gradients
        dists_to_origin = np.hypot(pts[:, 0], pts[:, 1])
        boundary_viol = np.maximum(0.0, dists_to_origin - (1.0 - r_target))
        
        # Gradient w.r.t. points
        grad_pts = np.zeros_like(pts)
        for i in range(N):
            if boundary_viol[i] > 0:
                r_i = dists_to_origin[i]
                if r_i > 1e-12:
                    grad_pts[i, 0] = 2.0 * boundary_viol[i] * pts[i, 0] / r_i
                    grad_pts[i, 1] = 2.0 * boundary_viol[i] * pts[i, 1] / r_i
        
        # Pairwise gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        cutoff = 3.0 * r_target
        mask = (dists < cutoff) & (dists > 1e-12)
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dists[i, j]
                    viol = 2.0 * r_target - d
                    if viol > 0:
                        factor = 2.0 * viol / d
                        grad_pts[i, 0] -= factor * (pts[i, 0] - pts[j, 0])
                        grad_pts[i, 1] -= factor * (pts[i, 1] - pts[j, 1])
                        grad_pts[j, 0] += factor * (pts[i, 0] - pts[j, 0])
                        grad_pts[j, 1] += factor * (pts[i, 0] - pts[j, 1])
        
        # Backpropagate through network
        # grad_out = grad_pts reshaped
        grad_out = grad_pts.reshape(-1)  # (N*2,)
        
        # grad_b2 = grad_out
        grad_b2 = grad_out
        
        # grad_w2 = h^T @ grad_out
        grad_w2 = np.outer(h, grad_out)  # (hidden, N*2)
        
        # grad_h = w2_flat @ grad_out
        grad_h = w2_flat.T @ grad_out  # (hidden,)
        
        # grad_lat = W1 @ (grad_h * h_prime)
        grad_lat = W1.T @ (grad_h * h_prime)  # (latent_dim,)
        
        # Pack gradients
        grad = np.zeros_like(params)
        grad[:latent_dim] = grad_lat
        grad[latent_dim:latent_dim + hidden_dim * N * 2] = grad_w2.flatten()
        grad[latent_dim + hidden_dim * N * 2:] = grad_b2
        
        return grad
    
    # Initial parameters
    init_params = np.concatenate([latent, W2.flatten(), b2])
    
    best_score = _score(points_init)
    best_points = points_init.copy()
    
    # Optimize with L-BFGS-B
    result = minimize(
        objective,
        init_params,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
    )
    
    # Extract optimized points
    opt_params = result.x
    lat_opt = opt_params[:latent_dim]
    w2_opt = opt_params[latent_dim:latent_dim + hidden_dim * N * 2].reshape(hidden_dim, N * 2)
    b2_opt = opt_params[latent_dim + hidden_dim * N * 2:]
    
    h_opt = np.tanh(lat_opt @ W1 + b1)
    out_opt = (h_opt @ w2_opt + b2_opt).reshape(N, 2)
    
    s = _score(out_opt)
    if s > best_score:
        best_score = s
        best_points = out_opt.copy()
    
    return best_points, best_score


def construct():
    start_time = time.time()

    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)

    best_points = points.copy()
    best_score = _score(points)

    # Try NN-based optimization
    remaining = DEADLINE - (time.time() - start_time) - 10.0
    if remaining > 10.0:
        nn_limit = min(remaining * 0.7, 80.0)
        nn_points, nn_score = _optimize_nn(points, nn_limit)
        if nn_score > best_score:
            best_score = nn_score
            best_points = nn_points.copy()

    # Final validity check
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)

    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6

    return best_points
