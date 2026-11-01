import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm

def _covering_radius(points, test_points):
    # points: (N, 3)
    # test_points: (M, 3)
    # Compute angular distances from each test point to each of the N points
    # Use the dot product to compute cosines of angles
    cos_angles = np.einsum('ij,kj->ik', test_points, points)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    # For each test point, find the minimum angle to any of the N points
    min_angles = np.min(angles, axis=1)
    # The covering radius is the maximum of these minimum angles
    return np.max(min_angles)

def _soft_max(x, beta=100.0):
    # Approximate max using log-sum-exp
    x_max = np.max(x)
    return x_max + np.log(np.mean(np.exp(beta * (x - x_max)))) / beta

def _objective(x):
    points = _normalize(x.reshape(N, 3))
    # Use a dense grid of test points
    # For efficiency, use a fixed set of test points
    return _soft_max(_covering_radius(points, _test_points), beta=100.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()

    # Generate a dense set of test points on the sphere
    # Use a spherical grid with sufficient resolution
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi_grid)
    test_points = np.column_stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid)
    ]).flatten().reshape(-1, 3)
    test_points = _normalize(test_points)

    global _test_points
    _test_points = test_points

    best_x = x0.copy()
    best_obj = _objective(x0)

    # Optimize using L-BFGS-B
    result = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-8})
    
    if time.time() > deadline:
        return _normalize(best_x.reshape(N, 3))
    
    if result.fun < best_obj:
        best_x = result.x
        best_obj = result.fun

    # Final normalization and return
    return _normalize(best_x.reshape(N, 3))
