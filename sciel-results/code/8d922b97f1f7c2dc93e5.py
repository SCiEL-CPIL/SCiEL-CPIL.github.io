import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad_sphere(pts):
    """
    Compute energy and gradient for points ON the unit sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy (float), grad (N, 3) array tangent to the sphere.
    """
    # pts are assumed normalized, but ensure it for safety
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij)
    # d_ij = sqrt(dist_sq_ij)
    # log(1/d_ij) = -0.5 * log(dist_sq_ij)
    # Sum over i<j
    iu = np.triu_indices(N, k=1)
    log_dists = -0.5 * np.log(dist_sq[iu])
    energy = np.sum(log_dists)
    
    # Gradient: 
    # E = sum_{i<j} -log(d_ij)
    # dE/dp_i = sum_{j!=i} dE/dd_ij * dd_ij/dp_i
    # dE/dd_ij = -1/d_ij
    # dd_ij/dp_i = (p_i - p_j) / d_ij
    # So dE/dp_i = sum_{j!=i} (-1/d_ij) * (p_i - p_j) / d_ij = - sum_{j!=i} (p_i - p_j) / d_ij^2
    # = sum_{j!=i} (p_j - p_i) / d_ij^2
    
    # Vectorized:
    # inv_dist_sq = 1/d_ij^2 for i!=j
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j (p_j - p_i) * inv_dist_sq[i,j]
    # = sum_j p_j * inv_dist_sq[i,j] - p_i * sum_j inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space: grad - (grad . p_i) p_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad

def _energy_and_grad_flat(x_flat):
    """
    Wrapper for L-BFGS-B which operates on flat vectors in R^(3N).
    x_flat: (3N,) array.
    Reshapes to (N,3), normalizes, computes sphere energy/grad, 
    then back-projects gradient to R^3 coordinates.
    """
    pts = x_flat.reshape(N, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    p = pts / norms
    
    # Compute energy and gradient on sphere
    energy, grad_sphere = _energy_and_grad_sphere(p)
    
    # Back-project gradient to R^3
    # p = x / ||x||
    # dE/dx = dE/dp * dp/dx
    # dp/dx = (I - p p^T) / ||x||
    # So dE/dx = (grad_sphere - (grad_sphere . p) p) / ||x||
    # Note: grad_sphere is already tangent, so grad_sphere . p = 0.
    # Thus dE/dx = grad_sphere / ||x||
    
    grad_flat = grad_sphere / norms
    grad_flat = grad_flat.flatten()
    
    return energy, grad_flat

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _geodesic_dome(n):
    # Icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
             
    # Subdivide
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
            # Add new edges
            edges.append((i, len(verts) + len(new_verts) - 1))
            edges.append((len(verts) + len(new_verts) - 1, j))
        verts = np.vstack([verts, np.array(new_verts)])
        
    if len(verts) > n:
        # Randomly select n points
        rng = np.random.RandomState(42)
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
        
    return verts

def _random_points(n, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    initializers = [
        _fibonacci_spiral,
        _geodesic_dome,
        lambda n: _random_points(n, seed=123),
        lambda n: _random_points(n, seed=456),
        lambda n: _random_points(n, seed=789)
    ]
    
    best_energy = float('inf')
    best_points = None
    
    for init_func in initializers:
        if time.time() > deadline:
            break
            
        try:
            x0 = init_func(N)
            x0_flat = x0.flatten()
            
            # L-BFGS-B optimization
            # maxiter is a hard cap, but we also check time inside the loop? 
            # SciPy minimize doesn't have a direct time limit callback in all versions.
            # We will use a conservative maxiter and rely on the outer loop.
            # 150 points -> 450 variables. L-BFGS is fast.
            # Let's allow a few hundred iterations per start.
            
            res = minimize(
                _energy_and_grad_flat, 
                x0_flat, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x.reshape(N, 3)
            # Normalize final points
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            x_opt = x_opt / norms
            
            energy_opt, _ = _energy_and_grad_sphere(x_opt)
            
            if energy_opt < best_energy:
                best_energy = energy_opt
                best_points = x_opt.copy()
                
        except Exception:
            continue
            
    # If no optimization succeeded or we didn't run any, return Fibonacci
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        
    return best_points

# EVOLVE-BLOCK-END
