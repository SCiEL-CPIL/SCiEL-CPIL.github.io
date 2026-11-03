"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill

def _energy_and_grad(pts):
    """Compute log-energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    Gradient for point i: -sum_{j!=i} (xi - xj)/|xi - xj|^2
    """
    n = pts.shape[0]
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j != i} (xi - xj) / dist_ij^2
    # Note: (xi - xj) / dist^2 = diff[i,j] / dist_sq[i,j]
    # But we need to be careful: the gradient of -log(dist_ij) w.r.t. xi is -(xi - xj)/dist_ij^2
    # So grad_i = -sum_{j!=i} (xi - xj) / dist_ij^2
    # Wait: d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    # So the gradient contribution to point i from pair (i,j) is -(xi-xj)/dist_ij^2
    # Summing over all j != i:
    # grad_i = -sum_{j!=i} (xi - xj) / dist_ij^2
    
    # diff[i,j] = xi - xj
    # We want sum over j != i of diff[i,j] / dist_sq[i,j]
    # Then negate it
    inv_dist_sq = 1.0 / dist_sq  # (n, n), diagonal is 1.0 (but we'll zero it out)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy(pts):
    """Compute the logarithmic energy."""
    e, _ = _energy_and_grad(pts)
    return e

def _make_geodesic_dome(subdivisions=3):
    """Create a geodesic dome (subdivided icosahedron) with N points."""
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        
        def midpoint(v1, v2):
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_midpoints:
                mid = (verts[v1] + verts[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_new = np.vstack([verts, mid])
                verts[:] = verts_new
                edge_midpoints[key] = len(verts) - 1
            return edge_midpoints[key]
        
        for f in faces:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        
        faces = new_faces
    
    # We now have more than N vertices. We need to select N of them.
    # For now, just take the first N (they should be well-distributed)
    if len(verts) >= N:
        return verts[:N]
    else:
        # Fallback: Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi_angle = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    
    def time_left():
        return 150.0 - (time.time() - start_time) - DEADLINE_MARGIN
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy: multiple restarts with different initializations, then L-BFGS-B
    # with tangent-space projection
    
    initializations = []
    
    # 1. Geodesic dome
    initializations.append(_make_geodesic_dome(subdivisions=2))
    
    # 2. Fibonacci sphere
    initializations.append(_fibonacci_sphere(N))
    
    # 3. Random points (seeded for reproducibility)
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts)
    
    # 4. Geodesic dome with different subdivision
    initializations.append(_make_geodesic_dome(subdivisions=3))
    
    # 5. Another random
    rng = np.random.RandomState(789)
    pts = rng.randn(N, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    initializations.append(pts)
    
    for init_idx, init_pts in enumerate(initializations):
        if time_left() < 10.0:
            break
        
        # Optimize using L-BFGS-B on the 3D coordinates, then project back
        # We'll use a wrapper that projects to the sphere after each step
        # Actually, let's use a penalty or just project and use the gradient
        
        # Better approach: use scipy minimize with the constraint that points are on the sphere
        # We'll minimize in 3D but project back to the sphere
        # The gradient in the 3D space needs to be projected to the tangent space
        
        def objective_and_grad(x):
            pts = x.reshape(N, 3)
            # Project to sphere
            pts_norm = _project_to_sphere(pts)
            e, g = _energy_and_grad(pts_norm)
            # Project gradient to tangent space: g_tangent = g - (g·x̂)x̂
            x_hat = pts_norm
            g_tangent = g - np.sum(g * x_hat, axis=1, keepdims=True) * x_hat
            return e, g_tangent.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                init_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            pts_opt = _project_to_sphere(result.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(pts_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best result
    if best_pts is not None and time_left() > 20.0:
        def objective_and_grad(x):
            pts = x.reshape(N, 3)
            pts_norm = _project_to_sphere(pts)
            e, g = _energy_and_grad(pts_norm)
            x_hat = pts_norm
            g_tangent = g - np.sum(g * x_hat, axis=1, keepdims=True) * x_hat
            return e, g_tangent.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                best_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            pts_polished = _project_to_sphere(result.x.reshape(N, 3))
            e_polished, _ = _energy_and_grad(pts_polished)
            
            if e_polished < best_energy:
                best_pts = pts_polished
                best_energy = e_polished
        except Exception:
            pass
    
    # Ensure we return something valid
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final validation: check for very close points
    # If any points are too close, nudge them
    for _ in range(10):
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        min_dist_sq = np.min(dist_sq)
        if min_dist_sq > 1e-18:
            break
        # Nudge a random point
        i = np.random.randint(N)
        best_pts[i] += 0.01 * np.random.randn(3)
        best_pts[i] = best_pts[i] / np.linalg.norm(best_pts[i])
    
    return _project_to_sphere(best_pts)
