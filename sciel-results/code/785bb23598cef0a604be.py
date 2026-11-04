"""Rewritten program for minimal logarithmic energy of 150 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import math

N = 150


def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Define the negative logarithmic energy to maximize (or minimize negative)
    # Energy = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    # We want to minimize energy, so we minimize -energy = sum_{i<j} log(|xi - xj|)
    # But since we're minimizing, let's directly minimize the energy function.

    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize each point to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms

        # Compute all pairwise distances
        # diff[i,j] = x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        # Mask diagonal and upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d = dists[mask]

        # Clamp to avoid log(0)
        d = np.maximum(d, 1e-15)

        # Energy = sum log(1/d) = -sum log(d)
        energy = -np.sum(np.log(d))

        # Gradient: dE/dx_i = sum_{j != i} (1/|x_i - x_j|) * (x_i - x_j)/|x_i - x_j|^2
        # = sum_{j != i} (x_i - x_j) / |x_i - x_j|^3
        grad = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d_ij = np.linalg.norm(x[i] - x[j])
                if d_ij < 1e-15:
                    continue
                grad[i] += (x[i] - x[j]) / (d_ij ** 3)

        # Account for normalization constraint (Lagrange multiplier approach)
        # The gradient of the constraint g(x) = x.x - 1 is 2x
        # Project gradient onto tangent space: grad_proj = grad - (grad.x)*x
        for i in range(N):
            proj = np.dot(grad[i], x[i])
            grad[i] -= proj * x[i]

        return energy, grad.reshape(-1)

    # Use L-BFGS-B with projected gradient approach
    # We'll use a custom optimization that keeps points on the sphere

    best_x = x0.copy()
    best_energy = float('inf')

    # Run optimization with L-BFGS-B
    def objective(x_flat):
        e, g = energy_and_grad(x_flat)
        return e, g

    # Use scipy's minimize with L-BFGS-B
    result = minimize(
        objective,
        x0.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
    )

    if result.success or result.fun < best_energy:
        x_opt = result.x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        x_opt = x_opt / norms
        # Compute energy
        diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d = dists[mask]
        d = np.maximum(d, 1e-15)
        e = -np.sum(np.log(d))
        if e < best_energy:
            best_energy = e
            best_x = x_opt

    # Additional local refinement with smaller steps
    # Use a simple gradient descent with adaptive step size
    x = best_x.copy()
    lr = 0.01
    for it in range(2000):
        e, g = energy_and_grad(x.flatten())
        g = g.reshape(N, 3)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()
        x = x - lr * g
        # Normalize
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95

    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms

    return best_x
