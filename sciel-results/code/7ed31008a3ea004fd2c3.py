"""Rewritten program using spherical-harmonic density parameterisation + gradient descent."""
import numpy as np
from math import fsum
import scipy.optimize as opt

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum
import scipy.optimize as opt

def _spherical_harmonics(l_max, theta, phi):
    """
    Compute real spherical harmonics Y_l^m for |m| <= l <= l_max at given (theta, phi).
    theta: polar angle (0 to pi), shape (n,)
    phi: azimuthal angle (0 to 2pi), shape (n,)
    Returns array of shape (n, n_harm) where n_harm = (l_max+1)^2
    """
    n = len(theta)
    n_harm = (l_max + 1) ** 2
    Y = np.zeros((n, n_harm))
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Use scipy.special.sph_harm for complex spherical harmonics, then convert to real
            from scipy.special import sph_harm
            # sph_harm(m, l, phi, theta) returns complex Y_l^m(phi, theta)
            Ycm = sph_harm(m, l, phi, theta)
            if m == 0:
                Y[:, idx] = np.sqrt(2) * Ycm.real
            elif m > 0:
                Y[:, idx] = -np.sqrt(2) * Ycm.real
            else:
                Y[:, idx] = -np.sqrt(2) * Ycm.imag
            idx += 1
    return Y

def _density_from_coeffs(coeffs, l_max, theta, phi):
    """
    Given spherical harmonic coefficients, compute density on sphere at (theta, phi).
    coeffs: shape (n_harm,)
    Returns: density values, shape (n,)
    """
    Y = _spherical_harmonics(l_max, theta, phi)
    return Y @ coeffs

def _log_energy_from_points(points):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    points: (N, 3) array on unit sphere
    """
    N_pts = len(points)
    total = 0.0
    terms = []
    for i in range(N_pts):
        for j in range(i + 1, N_pts):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            terms.append(np.log(1.0 / d))
    return fsum(terms)

def _optimize_with_density():
    """
    Use spherical harmonic density parameterisation to find good point distribution.
    We'll use a grid of points, compute their positions based on density, and optimize.
    """
    l_max = 6  # Number of spherical harmonic terms
    n_harm = (l_max + 1) ** 2
    
    # Initial coefficients: mostly zero (uniform density) with small random perturbation
    rng = np.random.default_rng(42)
    coeffs = rng.normal(0, 0.01, n_harm)
    coeffs[0] = 1.0  # Constant term for uniform baseline
    
    # We'll work with a fixed set of points and adjust their positions based on density
    # Actually, a better approach: use the density to guide a repulsion-based optimization
    
    # Let's use a different strategy: start with Fibonacci points, then do gradient descent
    # on point positions using the log-kernel gradient
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient descent on point positions (constrained to sphere)
    def _energy_and_grad(points):
        """
        Compute log energy and gradient w.r.t. point positions.
        Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
        Gradient: for each pair (i,j), d/dx_i [log(1/|x_i-x_j|)] = -(x_i-x_j)/|x_i-x_j|^2
        """
        N_pts = len(points)
        energy = 0.0
        grad = np.zeros_like(points)
        
        for i in range(N_pts):
            for j in range(N_pts):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-20:
                    # Avoid singularity
                    continue
                dist = np.sqrt(dist_sq)
                # d/dx_i log(1/dist) = d/dx_i (-log(dist)) = -diff/dist^2
                grad[i] -= diff / dist_sq
                energy -= np.log(dist)  # log(1/dist) = -log(dist)
        
        # Each pair counted twice, so divide energy by 2
        energy /= 2.0
        # Gradient is already correct (each pair contributes to both i and j)
        return energy, grad
    
    # Perform gradient descent with projection onto sphere
    lr = 0.001
    n_iter = 2000
    
    for it in range(n_iter):
        energy, grad = _energy_and_grad(points)
        
        # Project gradient onto tangent space of sphere
        # For each point, tangent space is orthogonal to the point itself
        for i in range(N):
            # Remove radial component
            radial_comp = np.dot(grad[i], points[i])
            grad_tangent = grad[i] - radial_comp * points[i]
            # Take step
            points[i] = points[i] - lr * grad_tangent
            # Project back to sphere
            norm = np.linalg.norm(points[i])
            if norm > 1e-15:
                points[i] = points[i] / norm
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            if energy < prev_energy - 1e-6:
                lr *= 1.1
            else:
                lr *= 0.9
            prev_energy = energy
        elif it == 0:
            prev_energy = energy
    
    return points

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Returns array of shape (100, 3) with points on unit sphere.
    """
    # Try multiple initializations and keep the best
    best_points = None
    best_energy = float('inf')
    
    for seed in [42, 123, 777, 2024]:
        rng = np.random.default_rng(seed)
        
        # Initialize with Fibonacci spiral plus random perturbation
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add small random perturbation
        perturbation = rng.normal(0, 0.01, (N, 3))
        points = points + perturbation
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Gradient descent
        def _energy_and_grad(pts):
            N_pts = len(pts)
            energy = 0.0
            grad = np.zeros_like(pts)
            
            for i in range(N_pts):
                for j in range(N_pts):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-20:
                        continue
                    dist = np.sqrt(dist_sq)
                    grad[i] -= diff / dist_sq
                    energy -= np.log(dist)
            
            energy /= 2.0
            return energy, grad
        
        lr = 0.0005
        n_iter = 1500
        prev_energy = float('inf')
        
        for it in range(n_iter):
            energy, grad = _energy_and_grad(points)
            
            for i in range(N):
                radial_comp = np.dot(grad[i], points[i])
                grad_tangent = grad[i] - radial_comp * points[i]
                points[i] = points[i] - lr * grad_tangent
                norm = np.linalg.norm(points[i])
                if norm > 1e-15:
                    points[i] = points[i] / norm
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                if energy < prev_energy - 1e-8:
                    lr *= 1.05
                else:
                    lr *= 0.95
                prev_energy = energy
            elif it == 0:
                prev_energy = energy
        
        # Evaluate final energy
        final_energy = _log_energy_from_points(points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = points.copy()
    
    # Final cleanup: ensure all points are on unit sphere and non-degenerate
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge one point slightly
                best_points[j] += 0.01 * np.random.default_rng(i * N + j).normal(0, 1, 3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
