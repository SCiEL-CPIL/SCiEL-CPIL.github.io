import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    # Note: log(1/d) = -log(d). d = sqrt(d_sq). log(d) = 0.5 log(d_sq).
    # So term is -0.5 log(d_sq).
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _objective_and_grad(x_flat):
    """
    Objective and gradient for L-BFGS-B.
    x_flat: flattened (N, 3) array.
    """
    pts = x_flat.reshape(N, 3)
    
    # Normalize points to ensure they stay on the sphere during optimization steps
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    e = _log_energy(pts)
    if not np.isfinite(e):
        return e, np.zeros_like(x_flat)
        
    # Compute gradient on the sphere
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances again for safety
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return e, np.zeros_like(x_flat)

    # Euclidean gradient of the energy with respect to pts
    # E = -0.25 * sum_{i,j} log(d_ij^2)
    # dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # The optimization is in the ambient R^3N space, but we project to sphere.
    # If we treat the input x_flat as unconstrained and project it to pts,
    # the chain rule applies: dE/dx_flat = (d pts / d x_flat)^T * (dE/dpts)
    # Since pts = x_flat / ||x_flat||, the gradient in x_flat space that,
    # when projected to the tangent space, matches grad_eucl, is simply grad_eucl
    # if we assume the optimizer moves in the tangent direction.
    # However, L-BFGS-B works in the flat space.
    # A standard trick for sphere constraints in flat optimizers is to use the
    # Euclidean gradient directly, as the retraction (normalization) is an
    # isometry on the tangent space to first order.
    # Actually, dE/dx_flat = grad_eucl - (grad_eucl . pts) pts is the Riemannian grad.
    # But since we are passing x_flat which is normalized to pts,
    # the gradient of E(pts(x_flat)) w.r.t x_flat is:
    # sum_k (d pts_k / d x_flat_k) * grad_eucl_k
    # This is effectively projecting the gradient back to the ambient space.
    # For simplicity and robustness with L-BFGS-B, using the Euclidean gradient
    # computed on the normalized points is a common and effective heuristic,
    # especially since we re-normalize at every step.
    
    # Let's verify: if we move x_flat by -alpha * grad, and then normalize,
    # the movement on the sphere is approximately -alpha * (grad - (grad.p)p).
    # So returning grad_eucl is consistent with the geometry if the optimizer
    # steps are small.
    
    return e, grad_eucl.flatten()

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    # Level 2 gives 162 points. We want 150.
    verts, _ = _icosa_subdivide(verts, faces, 2)
    
    if len(verts) > n_points:
        # Remove points that are closest to their nearest neighbor
        min_dists = np.zeros(len(verts))
        for i in range(len(verts)):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            min_dists[i] = np.min(dists)
        
        order = np.argsort(min_dists)
        indices_to_keep = order[len(verts)-n_points:]
        verts = verts[indices_to_keep]
        
    return verts

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * math.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron) - Primary strategy
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere - Fallback / diverse start
    inits.append(_fibonacci_sphere(N))
    
    # 3. Random Uniform - Diverse start
    for seed in [42, 123, 999]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # Optimization Loop using L-BFGS-B
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        try:
            x0 = init_pts.flatten()
            # Use L-BFGS-B for robustness
            # maxiter is capped, but we also check time
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = res.x.reshape(N, 3)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do a final polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            opt_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
