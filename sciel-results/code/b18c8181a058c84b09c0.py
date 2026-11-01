import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 3
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    # Small random perturbation to break symmetry
    points += np.random.normal(0, s * 0.005, size=points.shape)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta, penalty_weight=100.0):
    """
    Compute the negative soft-min value with penalty for constraint violations.
    """
    C = x.reshape(N, 2)
    
    # 1. Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Extract pair distances
    L_flat = L[i_idx, j_idx]
    
    # Handle zero distances for gradient stability
    eps = 1e-8
    L_safe = np.maximum(L_flat, eps)
    
    x_pairs_flat = L_safe / 2.0
    
    # Combine all x values
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Compute soft-min using log-sum-exp trick
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    # Value: S = min_x - (1/beta) * log(Z)
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Add penalty for points outside the unit circle
    outside = np.maximum(0.0, L_wall - 1.0)
    penalty = penalty_weight * np.sum(outside ** 2)
    
    # Gradient of S with respect to C
    grad = np.zeros_like(C)
    
    # Pair contributions
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    # U = (C_i - C_j) / L
    U = D_flat / L_safe[:, np.newaxis]
    
    # dx_pair/dC_i = (C_i - C_j) / L
    # dS/dx_pair = P_pairs
    # dS/dC_i = sum_j P_ij * d(x_ij)/dC_i = sum_j P_ij * (C_i - C_j) / L_ij * 0.5? 
    # Wait, x_pair = L/2. d(x_pair)/dC_i = (C_i - C_j)/L * 0.5? No.
    # L = ||C_i - C_j||. dL/dC_i = (C_i - C_j)/L.
    # x = L/2. dx/dC_i = 0.5 * (C_i - C_j)/L.
    # dS/dC_i = sum_j P_ij * 0.5 * (C_i - C_j)/L_ij
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall contributions
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.maximum(L_wall, eps)
    # x_wall = 1 - ||C_i||
    # dx_wall/dC_i = -C_i / ||C_i||
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    # Penalty gradient
    # Penalty = w * sum(outside^2) where outside = max(0, ||C_i|| - 1)
    # dPenalty/dC_i = 2*w*outside * C_i/||C_i||
    penalty_grad = 2.0 * penalty_weight * outside[:, np.newaxis] * C / L_wall_safe[:, np.newaxis]
    grad += penalty_grad
    
    # We want to minimize -S + penalty, so gradient is -grad_S + penalty_grad
    return -(val - penalty), -(grad - penalty_grad)

def _enforce_constraints(points, r_margin=0.0):
    """Ensure points are inside the unit circle with a small margin."""
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        if norms[i] > 1.0 - r_margin:
            if norms[i] > 1e-10:
                points[i] = points[i] * (1.0 - r_margin) / norms[i]
            else:
                points[i] = np.array([1.0 - r_margin, 0.0])
    return points

def _ring_init(n_points):
    """Generate points in concentric rings inside the unit circle."""
    points = []
    points.append([0.0, 0.0])
    
    ring = 1
    while len(points) < n_points:
        n_ring = 6 * ring
        if len(points) + n_ring > n_points:
            n_ring = n_points - len(points)
        
        r = ring * np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        if r > 1.0:
            r = 1.0 - 0.01
        
        for k in range(n_ring):
            angle = 2 * np.pi * k / n_ring + np.pi / ring
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
    
    return np.array(points[:n_points])

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    random_pts = np.random.randn(N, 2)
    random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
    random_pts *= 0.9
    inits.append(random_pts)
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        init_points = _enforce_constraints(init_points)
        init_score = _compute_true_score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_true_score(best_points)
    
    current_points = best_points.copy()
    
    # Beta schedule: Start low, increase gradually, then high for polish
    beta = 10.0
    max_beta = 5000.0
    beta_factor = 1.4
    
    while time.time() - start_time < time_limit - 5.0:
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            # Limit iterations to avoid hanging in one step
            maxiter = min(500, max(50, int(remaining * 2)))
            
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_points = res.x.reshape(N, 2)
                new_points = _enforce_constraints(new_points)
                new_score = _compute_true_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                
                current_points = new_points
        except Exception:
            pass
            
        beta *= beta_factor
        if beta > max_beta:
            break

    # Final high-precision polish
    # Use very high beta and tight tolerances
    for final_beta in [max_beta, max_beta * 2, max_beta * 5]:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        def final_obj_and_grad(x, fb=final_beta):
            return _soft_min_obj_and_grad(x, fb, penalty_weight=10.0)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            maxiter = min(1000, max(100, int(remaining * 5)))
            res = minimize(final_obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100})
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_points = res.x.reshape(N, 2)
                new_points = _enforce_constraints(new_points)
                new_score = _compute_true_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                current_points = new_points
        except Exception:
            pass

    # Final local refinement: try small perturbations around the best solution
    if time.time() - start_time < time_limit - 1.0:
        for _ in range(5):
            if time.time() - start_time > time_limit - 0.5:
                break
            # Try a small random perturbation
            perturbed = best_points + np.random.normal(0, 0.0005, size=best_points.shape)
            perturbed = _enforce_constraints(perturbed)
            
            def final_polish(x):
                return _soft_min_obj_and_grad(x, 10000.0, penalty_weight=5.0)
            
            try:
                res = minimize(final_polish, perturbed.flatten(), method='L-BFGS-B', jac=True, 
                               options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    new_points = res.x.reshape(N, 2)
                    new_points = _enforce_constraints(new_points)
                    new_score = _compute_true_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
    
    return best_points
