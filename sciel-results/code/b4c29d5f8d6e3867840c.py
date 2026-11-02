import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    num_iterations = 3000
    # Start with a broad sigma to allow large movements, then narrow to refine
    # sigma is the width of the Gaussian kernel in radians
    initial_sigma = 0.5 
    final_sigma = 0.1
    # Learning rate
    initial_lr = 0.05
    final_lr = 0.001
    
    for it in range(num_iterations):
        # Anneal sigma and learning rate
        t = it / (num_iterations - 1)
        sigma = initial_sigma * (final_sigma / initial_sigma) ** t
        lr = initial_lr * (final_lr / initial_lr) ** t
        
        # Compute pairwise angular distances
        # dot = points @ points.T
        # cos_d = np.clip(dot, -1.0, 1.0)
        # d = np.arccos(cos_d)
        
        # To avoid arccos instability and improve speed, work with cos_d directly or d
        # Let's use d for clarity, but vectorize.
        dot = np.clip(points @ points.T, -1.0, 1.0)
        d = np.arccos(dot)
        
        # Kernel: K(d) = exp(-d^2 / (2 * sigma^2))
        # We want to MINIMIZE sum K(d_ij) to push points apart.
        # Gradient of K w.r.t x_i:
        # dK/dd = -d / sigma^2 * K(d)
        # dd/dx_i = - (x_j - (x_i . x_j) x_i) / sin(d)
        # So grad_xi K = (-d / sigma^2 * K) * (-(x_j - dot_ij x_i) / sin_d)
        #               = (d * K / sigma^2) * (x_j - dot_ij x_i) / sin_d
        
        # We want to move x_i in direction -grad_xi E = - sum_j grad_xi K_ij
        # Force = - sum_j [ (d_ij * K_ij / sigma^2) * (x_j - dot_ij x_i) / sin_d_ij ]
        
        # Vectorized calculation
        # d is (N, N)
        # K is (N, N)
        K = np.exp(-d**2 / (2 * sigma**2))
        np.fill_diagonal(K, 0) # No self-interaction
        
        # dK_dd = -d / sigma^2 * K
        dK_dd = -d / (sigma**2) * K
        
        # sin_d
        sin_d = np.sin(d)
        # Avoid division by zero for close points
        sin_d_safe = np.where(sin_d < 1e-6, 1e-6, sin_d)
        
        # grad_d = dK_dd
        # grad_xi_j = grad_d * (-(x_j - dot_ij x_i) / sin_d)
        # Note: (x_j - dot_ij x_i) is the component of x_j orthogonal to x_i.
        # Let v_ij = x_j - dot_ij x_i
        # grad_xi_j = dK_dd * (-v_ij / sin_d)
        
        # We need sum_j grad_xi_j
        # v_ij matrix: V[i,j] = points[j] - dot[i,j] * points[i]
        # This is a (N, N, 3) tensor.
        
        # Efficient computation:
        # V = points[None, :, :] - dot[:, :, None] * points[:, None, :]
        # V shape: (N, N, 3). V[i, j] is vector from x_i to projection of x_j?
        # Actually, x_j - (x_i . x_j) x_i is the tangent vector at x_i pointing towards x_j?
        # Yes, this is the direction of increasing d_ij?
        # Let's check: If x_j is close to x_i, x_j - dot x_i ~ x_j - x_i.
        # Gradient of d w.r.t x_i is - (x_j - dot x_i) / sin d.
        # So grad_xi K = dK_dd * (-(x_j - dot x_i) / sin d).
        
        V = points[None, :, :] - dot[:, :, None] * points[:, None, :]
        
        # grad_xi_j = dK_dd[:, :, None] * (-V / sin_d_safe[:, :, None])
        grad_xi_j = dK_dd[:, :, None] * (-V / sin_d_safe[:, :, None])
        
        # Sum over j (axis 1) to get gradient for each i
        grad_xi = np.sum(grad_xi_j, axis=1)
        
        # Update: x_i = x_i - lr * grad_xi
        points = points - lr * grad_xi
        
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms

    return points
