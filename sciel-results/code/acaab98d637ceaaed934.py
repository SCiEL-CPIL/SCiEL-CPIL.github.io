"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _build_knn_graph(pts, k=8):
    """Build a fixed k-nearest-neighbor connectivity graph from initial points."""
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    
    # For each point, find k nearest neighbors
    edges = set()
    for i in range(n):
        neighbors = np.argsort(dist_sq[i])[:k]
        for j in neighbors:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    edge_list = sorted(edges)
    return edge_list

def _compute_log_energy(pts, edge_list=None, lambda_reg=0.0):
    """
    Compute augmented logarithmic energy with optional edge-length regularization.
    E = -sum_{i<j} log(|xi - xj|) + lambda_reg * sum_{(i,j) in edges} (|xi - xj| - L_target)^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    
    # Logarithmic energy
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Edge-length regularization
    if edge_list is not None and lambda_reg > 0:
        # Target edge length: average distance from a reference configuration
        # Use the mean distance as target
        mean_dist = np.mean(d_upper)
        reg = 0.0
        for (i, j) in edge_list:
            d_ij = dist[i, j]
            reg += (d_ij - mean_dist) ** 2
        energy += lambda_reg * reg
    
    return energy

def _compute_augmented_grad(pts, edge_list=None, lambda_reg=0.0):
    """
    Compute gradient of augmented energy.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    # Gradient of log energy: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    # Gradient of edge-length regularization
    if edge_list is not None and lambda_reg > 0:
        dist = np.sqrt(dist_sq_safe)
        mean_dist = np.mean(dist[np.triu_indices(n, k=1)])
        
        for (i, j) in edge_list:
            d_ij = dist[i, j]
            if d_ij < 1e-15:
                continue
            # d/d(pts[i]) (d_ij - mean_dist)^2 = 2*(d_ij - mean_dist) * (pts[i] - pts[j]) / d_ij
            coeff = 2.0 * lambda_reg * (d_ij - mean_dist) / d_ij
            grad[i] += coeff * (pts[i] - pts[j])
            grad[j] -= coeff * (pts[i] - pts[j])
    
    return grad

def _riemannian_gradient_descent(pts, edge_list, max_iter=3000, lambda_reg=0.05, lr_init=0.01, lr_decay=0.999, tol=1e-10):
    """
    Riemannian gradient descent with adaptive learning rate on the sphere.
    Uses the augmented energy (log energy + edge regularization).
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _compute_log_energy(pts, edge_list, lambda_reg)
    current_grad = _compute_augmented_grad(pts, edge_list, lambda_reg)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    lr = lr_init
    best_energy = current_energy
    best_pts = pts.copy()
    patience = 0
    max_patience = 500
    
    for k in range(max_iter):
        # Adaptive learning rate: decrease if energy increases
        pts_new = pts - lr * grad_tangent
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _compute_log_energy(pts_new, edge_list, lambda_reg)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
            patience = 0
            # Slightly increase lr if improving
            lr = min(lr * 1.01, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.5
            patience += 1
            if patience > max_patience:
                break
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
        
        # Recompute gradient
        current_grad = _compute_augmented_grad(pts, edge_list, lambda_reg)
        grad_dot = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
        
        # Decay learning rate slowly
        lr *= lr_decay
    
    return best_pts, best_energy

def _pure_log_energy_and_grad(pts):
    """Compute pure logarithmic energy and its gradient (no regularization)."""
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

def _pure_log_energy_rcg(pts, max_iter=2000, tol=1e-10):
    """Riemannian CG for pure log energy (no regularization)."""
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _pure_log_energy_and_grad(pts)
    
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    d = -grad_tangent
    
    for k in range(max_iter):
        alpha = 0.1
        improved = False
        best_pts_new = pts.copy()
        best_energy = current_energy
        best_grad_new = current_grad.copy()
        
        for backtrack in range(40):
            pts_new = pts + alpha * d
            pts_new = _project_to_sphere(pts_new)
            new_energy, new_grad = _pure_log_energy_and_grad(pts_new)
            
            if new_energy < best_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7
            
            if alpha < 1e-10:
                break
        
        if not improved:
            break
        
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        
        if np.linalg.norm(grad_tangent) < tol:
            break
        
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = min(num / den, 1.0)
        
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        d = -grad_tangent_new + beta * d_tangent
        grad_tangent = grad_tangent_new
    
    return pts, current_energy

def construct():
    # Step 1: Generate initial configuration
    pts_init = _fibonacci_spiral(N, offset=0.5)
    
    # Step 2: Build fixed connectivity graph (k-NN)
    edge_list = _build_knn_graph(pts_init, k=8)
    
    # Step 3: Optimize with augmented energy (log + edge regularization)
    # Use multiple lambda values and take the best
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    for lambda_reg in [0.0, 0.01, 0.05, 0.1, 0.2]:
        try:
            res_pts, res_energy = _riemannian_gradient_descent(
                pts_init, edge_list, 
                max_iter=3000, lambda_reg=lambda_reg, 
                lr_init=0.01, lr_decay=0.999, tol=1e-10
            )
            # Evaluate with pure log energy
            pure_energy, _ = _pure_log_energy_and_grad(res_pts)
            if pure_energy < best_energy:
                best_energy = pure_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Step 4: Refine best solution with pure log energy RCG
    try:
        res_pts, res_energy = _pure_log_energy_rcg(best_pts, max_iter=3000, tol=1e-11)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts.copy()
    except Exception:
        pass
    
    # Step 5: Final fine-tuning with small perturbations
    for final_pass in range(3):
        try:
            noise_scale = 0.001 * (final_pass + 1)
            noise = np.random.normal(0, noise_scale, size=(N, 3))
            perturbed = _project_to_sphere(best_pts + noise)
            res_pts, res_energy = _pure_log_energy_rcg(perturbed, max_iter=2000, tol=1e-11)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final projection
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
