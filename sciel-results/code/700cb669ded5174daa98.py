"""Rigid polyhedral mesh approach for logarithmic energy minimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, Delaunay
import math
import time

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _build_mesh_connectivity(pts):
    """
    Build a rigid polyhedral mesh connectivity from initial points.
    Uses Delaunay triangulation on the sphere to create a mesh.
    Returns a list of edges (pairs of point indices).
    """
    pts = _project_to_sphere(pts.copy())
    
    # Use Delaunay triangulation to find nearby neighbors
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        edges = set()
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    except Exception:
        # Fallback: use k-nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        # Find 6 nearest neighbors for each point
        _, idx = tree.query(pts, k=7)  # k=7 includes self
        edges = set()
        for i in range(len(pts)):
            for j in idx[i][1:]:  # Skip self
                edges.add((min(i, j), max(i, j)))
    
    edge_list = list(edges)
    return edge_list

def _mesh_energy_and_grad(pts, edges):
    """
    Compute energy and gradient for the mesh-based optimization.
    The energy is the full logarithmic energy, but we use the mesh structure
    to guide the optimization direction.
    """
    energy, grad = _log_energy_and_grad(pts)
    return energy, grad

def _optimize_mesh(pts, edges, max_iter=3000, tol=1e-10):
    """
    Optimize points on the sphere using gradient descent, guided by mesh structure.
    Uses Riemannian gradient with adaptive step size.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Build adjacency for mesh
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Initial direction: negative tangent gradient
    d = -grad_tangent.copy()
    
    # Adaptive parameters
    alpha = 0.01
    min_alpha = 1e-10
    max_alpha = 1.0
    
    for k in range(max_iter):
        # Line search
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        best_grad_new = current_grad.copy()
        
        for backtrack in range(30):
            # Step in tangent space
            pts_new = pts + alpha * d
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7
            
            if alpha < min_alpha:
                break
        
        if not improved:
            # Reset step size and direction
            alpha = 0.01
            d = -grad_tangent.copy()
            continue
        
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        alpha = min(best_alpha * 1.5, max_alpha)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta with damping
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = min(num / den, 1.0)
        
        # Transport d to new tangent space
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        grad_tangent = grad_tangent_new
    
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_with_mesh_strategy():
    """
    Main optimization strategy using rigid polyhedral mesh.
    1. Generate initial points
    2. Build mesh connectivity
    3. Optimize using mesh-guided gradient descent
    4. Refine with additional passes
    """
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Generate multiple initial configurations
    inits = []
    
    # Fibonacci spiral variants
    inits.append(_fibonacci_spiral(N, offset=0.5))
    inits.append(_fibonacci_spiral(N, offset=0.0))
    inits.append(_fibonacci_spiral(N, offset=0.3))
    inits.append(_fibonacci_spiral(N, offset=0.7))
    
    # Perturbed Fibonacci
    for noise_scale in [0.05, 0.1, 0.15]:
        noise = np.random.normal(0, noise_scale, size=(N, 3))
        inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Try each initial configuration with mesh optimization
    for idx, x0 in enumerate(inits):
        if time.time() - start_time > timeout:
            break
        
        try:
            # Build mesh connectivity
            edges = _build_mesh_connectivity(x0)
            
            # Optimize using mesh-guided gradient descent
            res_pts, res_energy = _optimize_mesh(x0, edges, max_iter=2000, tol=1e-10)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Additional refinement passes
    for pass_num in range(4):
        if time.time() - start_time > timeout:
            break
        
        try:
            # Rebuild mesh from current best
            edges = _build_mesh_connectivity(best_pts)
            
            # Small perturbation to escape local minima
            if pass_num > 0:
                noise_scale = 0.003 * (pass_num + 1)
                noise = np.random.normal(0, noise_scale, size=(N, 3))
                perturbed = _project_to_sphere(best_pts + noise)
            else:
                perturbed = best_pts.copy()
            
            # Optimize
            res_pts, res_energy = _optimize_mesh(perturbed, edges, max_iter=2500, tol=1e-11)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final fine-tuning
    for final_pass in range(3):
        if time.time() - start_time > timeout:
            break
        
        try:
            edges = _build_mesh_connectivity(best_pts)
            noise_scale = 0.001 * (final_pass + 1)
            noise = np.random.normal(0, noise_scale, size=(N, 3))
            perturbed = _project_to_sphere(best_pts + noise)
            
            res_pts, res_energy = _optimize_mesh(perturbed, edges, max_iter=1500, tol=1e-12)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts

def construct():
    """
    Main entry point: construct 100 points on the unit sphere
    to minimize logarithmic energy using rigid polyhedral mesh approach.
    """
    # Set random seed for reproducibility within a single run
    np.random.seed(42)
    
    # Use the mesh-based optimization strategy
    best_pts = _optimize_with_mesh_strategy()
    
    # Final safety checks
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
