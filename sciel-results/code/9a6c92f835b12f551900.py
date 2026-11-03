"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    d = np.sqrt(d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    grad = M @ points
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - grad
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _project_gradient(points, grad):
    """Project gradient onto tangent space of unit sphere."""
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points


def _spherical_harmonic_basis(l_max, n_theta, n_phi):
    """
    Generate a basis of spherical harmonic-like functions on the sphere.
    We use a truncated Fourier series in phi and a polynomial in cos(theta)
    to approximate smooth functions.
    
    Returns a function that maps (theta, phi) to basis values.
    """
    # We'll use a simple spectral basis:
    # f(theta, phi) = sum_{l=0}^{l_max} sum_{m=-l}^{l} c_{l,m} * Y_{l,m}(theta, phi)
    # For simplicity, we'll use real spherical harmonics or just a Fourier series.
    
    # Actually, let's use a more practical approach:
    # Represent the point configuration as a set of "control points" in a spectral domain.
    # But for the energy minimization, we need to map from spectral coefficients to points.
    
    # Alternative: Use a grid of points and optimize their positions.
    # Let's stick with the direct optimization but use better initialization.
    
    # For the spectral approach, we can use the following:
    # The points are determined by the zeros of a certain function on the sphere.
    # This is complex to implement, so let's use a hybrid approach.
    
    return None


def _optimize_with_lbfgs(points, maxiter=200):
    """
    Optimize points on the sphere using L-BFGS-B.
    We parameterize each point by (theta, phi) in spherical coordinates.
    """
    # Convert from Cartesian to spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-15), -1, 1))  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    
    # Flatten to a single vector
    params = np.column_stack([theta, phi]).flatten()
    
    def objective(params):
        theta = params[0::2]
        phi = params[1::2]
        # Convert back to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        pts = _normalize(pts)
        return _log_energy(pts)
    
    def gradient(params):
        theta = params[0::2]
        phi = params[1::2]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        pts = _normalize(pts)
        
        # Compute gradient in Cartesian coordinates
        grad_cart = _log_energy_grad(pts)
        grad_cart = _project_gradient(pts, grad_cart)
        
        # Convert gradient to (theta, phi) coordinates
        # dE/dtheta = grad_cart dot d(pts)/dtheta
        # dE/dphi = grad_cart dot d(pts)/dphi
        
        # d(x,y,z)/dtheta = (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
        dx_dtheta = np.cos(theta) * np.cos(phi)
        dy_dtheta = np.cos(theta) * np.sin(phi)
        dz_dtheta = -np.sin(theta)
        
        # d(x,y,z)/dphi = (-sin(phi), cos(phi), 0)
        dx_dphi = -np.sin(phi)
        dy_dphi = np.cos(phi)
        dz_dphi = 0.0
        
        grad_theta = grad_cart[:, 0] * dx_dtheta + grad_cart[:, 1] * dy_dtheta + grad_cart[:, 2] * dz_dtheta
        grad_phi = grad_cart[:, 0] * dx_dphi + grad_cart[:, 1] * dy_dphi + grad_cart[:, 2] * dz_dphi
        
        # Interleave
        grad_params = np.zeros_like(params)
        grad_params[0::2] = grad_theta
        grad_params[1::2] = grad_phi
        
        return grad_params
    
    # Bounds for theta and phi
    bounds = []
    for i in range(N):
        bounds.append((0.0, np.pi))  # theta
        bounds.append((-np.pi, np.pi))  # phi
    
    result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Convert back to Cartesian
    theta_opt = result.x[0::2]
    phi_opt = result.x[1::2]
    x = np.sin(theta_opt) * np.cos(phi_opt)
    y = np.sin(theta_opt) * np.sin(phi_opt)
    z = np.cos(theta_opt)
    points_opt = np.column_stack([x, y, z])
    points_opt = _normalize(points_opt)
    
    return points_opt, result.fun


def _gradient_descent_sphere(points, lr, n_iters, momentum=0.9):
    """Gradient descent on the sphere with momentum."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        v = momentum * v - lr * grad / (grad_norm + 1e-15)
        points = points + v
        points = _normalize(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    return best_points, best_energy


def construct():
    # Start with multiple initializations
    best_points = None
    best_energy = float('inf')
    
    # Initialization 1: Fibonacci Spiral
    pts_fib = _fibonacci_spiral(N)
    pts_fib = _normalize(pts_fib)
    
    # Phase 1: Coarse gradient descent
    pts1, e1 = _gradient_descent_sphere(pts_fib, lr=0.005, n_iters=300, momentum=0.9)
    
    # Phase 2: L-BFGS-B optimization
    pts2, e2 = _optimize_with_lbfgs(pts1, maxiter=150)
    
    if e2 < best_energy:
        best_energy = e2
        best_points = pts2
    
    # Initialization 2: Random points
    rng = np.random.default_rng(42)
    for seed in range(3):
        pts_rand = _random_points(N, seed=100 + seed)
        
        # Coarse gradient descent
        pts_r1, e_r1 = _gradient_descent_sphere(pts_rand, lr=0.005, n_iters=200, momentum=0.9)
        
        # L-BFGS-B optimization
        pts_r2, e_r2 = _optimize_with_lbfgs(pts_r1, maxiter=100)
        
        if e_r2 < best_energy:
            best_energy = e_r2
            best_points = pts_r2
    
    # Final polish from best result
    if best_points is not None:
        pts_final, e_final = _optimize_with_lbfgs(best_points, maxiter=100)
        if e_final < best_energy:
            best_energy = e_final
            best_points = pts_final
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_points = _normalize(best_points)
    else:
        best_points = _normalize(best_points)
        
    # Verify no points are too close
    d = _pairwise_distances(best_points)
    if d.min() < 1e-9:
        best_points = _fibonacci_spiral(N)
        best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
