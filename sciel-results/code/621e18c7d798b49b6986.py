"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
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
    
    # Compute pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
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

def _riemannian_gradient_descent(pts, max_iter=5000, tol=1e-10, time_budget=30.0):
    """
    Perform Riemannian Gradient Descent on the sphere with adaptive step size.
    Uses a simple but robust approach: project gradient to tangent space,
    take step, project back to sphere.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Adaptive step size
    alpha = 0.001
    alpha_min = 1e-12
    alpha_max = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    for k in range(max_iter):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Take step in tangent space
        pts_new = pts + alpha * grad_tangent
        # Retraction (project to sphere)
        pts_new = _project_to_sphere(pts_new)
        
        new_energy, new_grad = _log_energy_and_grad(pts_new)
        
        if new_energy < current_energy:
            # Improved - increase step size slightly
            pts = pts_new
            current_energy = new_energy
            current_grad = new_grad
            no_improve_count = 0
            alpha = min(alpha * 1.05, alpha_max)
            
            # Project new gradient to tangent space
            grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
            grad_tangent = current_grad - grad_dot_new[:, np.newaxis] * pts
        else:
            # No improvement - decrease step size
            alpha *= 0.5
            no_improve_count += 1
            
            if alpha < alpha_min or no_improve_count > max_no_improve:
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
    
    return pts, current_energy

def _riemannian_conjugate_gradient(pts, max_iter=500, tol=1e-8, time_budget=20.0):
    """
    Perform Riemannian Conjugate Gradient on the sphere with improved line search.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Adaptive line search (Backtracking with Wolfe-like conditions)
        alpha = 0.1
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        best_grad_new = current_grad.copy()
        
        for backtrack in range(40):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7  # More aggressive backtracking
            
            if alpha < 1e-10:
                break
        
        if not improved:
            break
            
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        
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
            beta = num / den
            beta = min(beta, 1.0)  # Damp beta to avoid divergence
            
        # Transport d to new tangent space
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Update grad_tangent for next iteration's beta calculation
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_ratio_spiral(n=N, start_offset=0.0):
    """Generate points using golden ratio spacing with different starting offset."""
    i = np.arange(n) + start_offset
    z = 1.0 - 2.0 * (i + 0.5) / n
    golden = 2.39996322972865332  # 2*pi / phi^2
    phi_angle = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices
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

def _refine_with_lbfgs(pts, max_iter=5000, time_budget=30.0):
    """Refine points using L-BFGS-B in 3D with sphere constraint via penalty."""
    start_time = time.time()
    
    def penalty_energy(x):
        pts = x.reshape(-1, 3)
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
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
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_sq_safe = np.maximum(dist_sq, 1e-30)
        
        # Gradient of energy
        W = np.zeros_like(dist_sq_safe)
        np.fill_diagonal(W, 0.0)
        W = 1.0 / dist_sq_safe
        np.fill_diagonal(W, 0.0)
        
        row_sum = W.sum(axis=1)
        grad_energy = -pts * row_sum[:, np.newaxis] + W @ pts
        
        # Gradient of penalty
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        grad_penalty = 2e6 * (norms - 1.0) * pts / norms
        
        return (grad_energy + grad_penalty).flatten()
    
    x0 = pts.flatten()
    
    # Use a callback to check time budget
    stop_flag = [False]
    def time_check_callback(k):
        if time.time() - start_time > time_budget:
            stop_flag[0] = True
        return stop_flag[0]
    
    res = minimize(penalty_energy, x0, jac=penalty_grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
                   callback=time_check_callback)
    
    best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    return best_pts

def _perturb_and_refine(pts, noise_scale=0.01, max_iter=2000, time_budget=20.0):
    """Perturb points and refine using RCG."""
    noise = np.random.normal(0, noise_scale, size=(N, 3))
    perturbed = _project_to_sphere(pts + noise)
    res_pts, res_energy = _riemannian_conjugate_gradient(perturbed, max_iter=max_iter, tol=1e-10, time_budget=time_budget)
    return res_pts, res_energy

def _refine_with_hybrid(pts, time_budget=60.0):
    """
    Hybrid refinement: alternate between RCG and gradient descent,
    with occasional perturbations to escape local minima.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    energy, _ = _log_energy_and_grad(pts)
    
    # Phase 1: RCG for coarse convergence
    pts, energy = _riemannian_conjugate_gradient(pts, max_iter=1000, tol=1e-8, time_budget=time_budget * 0.4)
    
    # Phase 2: Gradient descent for fine tuning
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        pts, energy = _riemannian_gradient_descent(pts, max_iter=5000, tol=1e-10, time_budget=remaining_time * 0.5)
    
    # Phase 3: Small perturbations and refinement
    remaining_time = time_budget - (time.time() - start_time)
    for _ in range(3):
        if time.time() - start_time > time_budget:
            break
        noise_scale = 0.002
        perturbed = _project_to_sphere(pts + np.random.normal(0, noise_scale, size=(N, 3)))
        new_pts, new_energy = _riemannian_conjugate_gradient(perturbed, max_iter=500, tol=1e-9, time_budget=5.0)
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
    
    # Final gradient descent pass
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 3:
        pts, energy = _riemannian_gradient_descent(pts, max_iter=3000, tol=1e-10, time_budget=remaining_time)
    
    return pts, energy

def construct():
    """
    Main construction function that finds low-energy configurations of 100 points on the sphere.
    Uses multiple initialization strategies and hybrid optimization.
    """
    np.random.seed(42)
    start_time = time.time()
    total_budget = 110.0  # Leave 10s margin for safety
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Fibonacci spiral (standard)
    inits.append(_fibonacci_spiral(N, offset=0.5))
    
    # 2. Fibonacci spiral with different offset
    inits.append(_fibonacci_spiral(N, offset=0.0))
    
    # 3. Golden ratio spiral
    inits.append(_golden_ratio_spiral(N, start_offset=0.0))
    
    # 4. Golden ratio spiral with offset
    inits.append(_golden_ratio_spiral(N, start_offset=0.5))
    
    # 5. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 6. Icosahedral-based
    inits.append(_icosahedral_based_init(N))
    
    # 7. Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    # 8. Rotated Fibonacci
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    inits.append(_project_to_sphere(_fibonacci_spiral(N) @ rot.T))
    
    # 9. Another random perturbation
    noise2 = np.random.normal(0, 0.2, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise2))
    
    # 10. Scaled perturbation
    noise3 = np.random.normal(0, 0.05, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise3))
    
    # 11-15: More diverse random initializations
    for i in range(5):
        noise = np.random.normal(0, 0.15 + 0.05*i, size=(N, 3))
        inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Phase 1: Run RCG from each initial configuration with time budget
    per_init_budget = total_budget * 0.4 / len(inits)
    for idx, x0 in enumerate(inits):
        if time.time() - start_time > total_budget * 0.4:
            break
        try:
            res_pts, res_energy = _riemannian_conjugate_gradient(x0, max_iter=1500, tol=1e-8, time_budget=per_init_budget)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Hybrid refinement on best solution
    remaining = total_budget - (time.time() - start_time)
    if remaining > 10:
        try:
            res_pts, res_energy = _refine_with_hybrid(best_pts, time_budget=min(remaining * 0.6, 60.0))
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement
    remaining = total_budget - (time.time() - start_time)
    if remaining > 10:
        try:
            refined = _refine_with_lbfgs(best_pts, max_iter=3000, time_budget=min(remaining * 0.4, 30.0))
            refined_energy, _ = _log_energy_and_grad(refined)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_pts = refined
        except Exception:
            pass
    
    # Phase 4: Final fine-tuning with small perturbations
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5:
        for final_pass in range(3):
            if time.time() - start_time > total_budget - 2:
                break
            try:
                noise_scale = 0.001 * (final_pass + 1)
                res_pts, res_energy = _perturb_and_refine(best_pts, noise_scale=noise_scale, max_iter=1500, time_budget=5.0)
                if res_energy < best_energy:
                    best_energy = res_energy
                    best_pts = res_pts.copy()
            except Exception:
                pass
    
    # Final gradient descent pass with remaining time
    remaining = total_budget - (time.time() - start_time)
    if remaining > 3:
        try:
            res_pts, res_energy = _riemannian_gradient_descent(best_pts, max_iter=3000, tol=1e-10, time_budget=remaining)
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
