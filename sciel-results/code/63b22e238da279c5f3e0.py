"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq = dist_safe ** 2
    
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_conjugate_gradient(pts, max_iter=500, tol=1e-8):
    """
    Perform Riemannian Conjugate Gradient on the sphere.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Line search (Backtracking)
        alpha = 0.1
        improved = False
        for backtrack in range(30):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                pts = pts_new
                current_energy = new_energy
                current_grad = new_grad
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            break
            
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
            
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = num / den
            
        # Transport d to new tangent space
        # d is tangent to pts_old.
        # Project d onto tangent space of pts_new
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Update grad_tangent for next iteration's beta calculation
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices and subdivide
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add perturbed Fibonacci points
    fib = _fibonacci_spiral(n)
    noise = np.random.normal(0, 0.05, size=(n, 3))
    pts = fib + noise
    return _project_to_sphere(pts)

def _refine_with_lbfgs(pts, max_iter=5000):
    """Refine points using L-BFGS-B in 3D with sphere constraint via penalty."""
    def penalty_energy(x):
        pts = x.reshape(-1, 3)
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist_safe = np.maximum(dist, 1e-15)
        
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist_safe[iu]))
        
        # Penalty for being off the sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        
        return energy + penalty
    
    def penalty_grad(x):
        pts = x.reshape(-1, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist_safe = np.maximum(dist, 1e-15)
        
        # Gradient of energy
        dist_sq = dist_safe ** 2
        W = np.zeros_like(dist_sq)
        np.fill_diagonal(W, 0.0)
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        row_sum = W.sum(axis=1)
        grad_energy = -pts * row_sum[:, np.newaxis] + W @ pts
        
        # Gradient of penalty
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        grad_penalty = 2e6 * (norms - 1.0) * pts / norms
        
        return (grad_energy + grad_penalty).flatten()
    
    x0 = pts.flatten()
    res = minimize(penalty_energy, x0, jac=penalty_grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    return best_pts

def construct():
    # Multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 3. Icosahedral-based
    inits.append(_icosahedral_based_init(N))
    
    # 4. Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    # 5. Rotated Fibonacci
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    inits.append(_project_to_sphere(_fibonacci_spiral(N) @ rot.T))
    
    # 6. Another random perturbation
    noise2 = np.random.normal(0, 0.2, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise2))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Run Riemannian CG from each initial configuration
    for idx, x0 in enumerate(inits):
        try:
            res_pts, res_energy = _riemannian_conjugate_gradient(x0, max_iter=2000, tol=1e-9)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Additional refinement passes from best solution
    for pass_num in range(3):
        try:
            # Small perturbation to escape local minima
            if pass_num > 0:
                noise = np.random.normal(0, 0.01 * (pass_num + 1), size=(N, 3))
                perturbed = _project_to_sphere(best_pts + noise)
            else:
                perturbed = best_pts.copy()
            
            res_pts, res_energy = _riemannian_conjugate_gradient(perturbed, max_iter=3000, tol=1e-10)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B refinement
    try:
        refined = _refine_with_lbfgs(best_pts, max_iter=3000)
        refined_energy, _ = _log_energy_and_grad(refined)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined
    except Exception:
        pass
    
    # One more RCG pass after L-BFGS
    try:
        res_pts, res_energy = _riemannian_conjugate_gradient(best_pts, max_iter=2000, tol=1e-10)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts
    except Exception:
        pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
