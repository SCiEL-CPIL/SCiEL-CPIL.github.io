import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns (energy, gradient) for optimization.
    """
    n = len(pts)
    # Ensure points are on sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), None
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    return energy, grad_tangent

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy and tangent gradient
    energy, grad_tangent = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw = grad_tangent / norms
    
    return energy, grad_raw.flatten()

def _fibonacci_sphere(n, jitter=0.0, seed=0):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if jitter > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, jitter, size=(n, 3))
        pts = pts + noise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
    return pts

def _subdivided_icosahedron(n):
    """
    Generate points on a sphere by subdividing an icosahedron.
    This provides a more uniform initial distribution than Fibonacci for some N.
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Icosahedron vertices
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    
    # Normalize vertices
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    # We need to determine subdivision depth such that total points is close to n
    # Points = 10 * 4^k + 2
    # For n=150, k=2 gives 162 points. k=1 gives 42 points.
    # We'll subdivide to k=2 (162 points) and then select the best 150 or just use 162 and trim?
    # Trimming might break symmetry.
    # Actually, we can just use the 162 points and let the optimizer merge/close them if it wants, 
    # or we can just use the 162 points as the initial guess for N=150? 
    # No, we must have exactly N points.
    # Let's subdivide to get ~150.
    # k=1 -> 42. k=2 -> 162.
    # 162 is close to 150. We can generate 162 and then remove 12 points?
    # Or we can generate 162, optimize, and then if we have extra, we are stuck.
    # The problem requires exactly N points.
    # So we should generate exactly N points.
    # Icosahedron subdivision gives specific numbers.
    # Maybe a better approach: Start with Icosahedron (12 points), subdivide until > N, then randomly select N?
    # Or just use Fibonacci which is flexible.
    # Let's stick to Fibonacci for flexibility, but maybe add a "random" start.
    
    # For now, let's just return a Fibonacci sphere if the specific count doesn't match a subdivision level.
    # Or we can implement a general subdivision that allows picking N points from the generated set.
    
    # Let's try to generate a set of points slightly larger than N and pick N.
    # Or just use Fibonacci. It's robust.
    
    # Let's create a "random" start instead for diversity.
    rng = np.random.default_rng(123)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Define multiple initial strategies
    inits = []
    
    # 1. Fibonacci
    inits.append(_fibonacci_sphere(N))
    
    # 2. Fibonacci with jitter
    inits.append(_fibonacci_sphere(N, jitter=0.1, seed=1))
    inits.append(_fibonacci_sphere(N, jitter=0.05, seed=2))
    
    # 3. Random
    inits.append(_subdivided_icosahedron(N)) # Currently random
    inits.append(_subdivided_icosahedron(N))
    
    best_pts = None
    best_energy = float('inf')
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10: # Leave 10s for final check
            break
            
        try:
            # Optimize using L-BFGS-B
            # We use a relatively small maxiter to allow multiple restarts
            # or to finish quickly if it converges
            res = minimize(
                _objective_and_grad,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 100,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _log_energy_fast(pts_opt)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
                
        except Exception:
            pass
            
    # If we have time, do a final polish on the best one
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(
                _objective_and_grad,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _log_energy_fast(pts_opt)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
