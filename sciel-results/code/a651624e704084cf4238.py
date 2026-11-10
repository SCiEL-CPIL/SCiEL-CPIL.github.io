# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    n = 13
    # Precompute indices for all 286 triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    num_tris = len(tri_i)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_score = -np.inf
    best_points = None
    
    def compute_score(pts):
        """Compute the actual score: min_triangle_area / hull_area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return -np.inf, None
            
            p_i = pts[tri_i]
            p_j = pts[tri_j]
            p_k = pts[tri_k]
            areas = 0.5 * np.abs(
                p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
            )
            min_area = np.min(areas)
            score = min_area / hull_area
            return score, hull
        except Exception:
            return -np.inf, None
    
    # Initial configurations
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Hexagonal lattice arrangement (better for Heilbronn)
    # Place 7 points in a hexagonal pattern, plus 6 more
    hex_pts = []
    # Center
    hex_pts.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts.append([np.cos(angle), np.sin(angle)])
    # Second ring - add 6 more points at radius 2
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        hex_pts.append([2*np.cos(angle), 2*np.sin(angle)])
    hex_arr = np.array(hex_pts[:n])
    
    # 3. Grid-like arrangement
    grid_pts = []
    for i in range(4):
        for j in range(3):
            grid_pts.append([i - 1.5, j - 1])
    grid_arr = np.array(grid_pts[:n])
    
    # 4. Circle with interior points
    circle_interior = []
    # 7 points on circle
    for i in range(7):
        angle = i * 2 * np.pi / 7
        circle_interior.append([np.cos(angle), np.sin(angle)])
    # 6 interior points in smaller circle
    for i in range(6):
        angle = i * 2 * np.pi / 6 + np.pi/6
        circle_interior.append([0.4*np.cos(angle), 0.4*np.sin(angle)])
    circle_arr = np.array(circle_interior[:n])
    
    # List of initial points to try
    initializations = [base_poly.copy(), hex_arr.copy(), grid_arr.copy(), circle_arr.copy()]
    
    # Add random perturbations
    rng = np.random.default_rng(42)
    for init in initializations:
        for _ in range(3):
            noise = rng.normal(0, 0.05, (n, 2))
            initializations.append(init + noise)
    
    # Add fully random points
    for _ in range(5):
        initializations.append(rng.random((n, 2)) * 2 - 1)
    
    # Beta schedule for soft-min
    beta_schedule = [20, 100, 500, 2000, 10000]
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        current_pts = init_pts.copy()
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - safety_margin:
                break
                
            def obj_and_grad(p_flat):
                pts = p_flat.reshape(n, 2)
                
                # 1. Compute triangle areas
                p_i = pts[tri_i]
                p_j = pts[tri_j]
                p_k = pts[tri_k]
                
                signed_areas = 0.5 * (
                    p_i[:, 0] * (p_j[:, 1] - p_k[:, 1]) +
                    p_j[:, 0] * (p_k[:, 1] - p_i[:, 1]) +
                    p_k[:, 0] * (p_i[:, 1] - p_j[:, 1])
                )
                areas = np.abs(signed_areas)
                
                # Soft-min
                max_area = np.max(areas)
                exp_terms = np.exp(-beta * (areas - max_area))
                sum_exp = np.sum(exp_terms)
                
                if sum_exp < 1e-300:
                    return -1e10, np.zeros_like(p_flat)
                    
                soft_min_val = max_area - 1.0 / beta * np.log(sum_exp)
                
                # Gradient of soft_min_val wrt pts
                weights = exp_terms / sum_exp
                
                signs = np.sign(signed_areas)
                signs[signs == 0] = 1.0
                
                grad_soft_min = np.zeros_like(pts)
                
                d_dx_i = 0.5 * (p_j[:, 1] - p_k[:, 1])
                d_dy_i = 0.5 * (p_k[:, 0] - p_j[:, 0])
                
                d_dx_j = 0.5 * (p_k[:, 1] - p_i[:, 1])
                d_dy_j = 0.5 * (p_i[:, 0] - p_k[:, 0])
                
                d_dx_k = 0.5 * (p_i[:, 1] - p_j[:, 1])
                d_dy_k = 0.5 * (p_j[:, 0] - p_i[:, 0])
                
                w = weights * signs
                
                np.add.at(grad_soft_min, (tri_i, 0), w * d_dx_i)
                np.add.at(grad_soft_min, (tri_i, 1), w * d_dy_i)
                np.add.at(grad_soft_min, (tri_j, 0), w * d_dx_j)
                np.add.at(grad_soft_min, (tri_j, 1), w * d_dy_j)
                np.add.at(grad_soft_min, (tri_k, 0), w * d_dx_k)
                np.add.at(grad_soft_min, (tri_k, 1), w * d_dy_k)
                
                # 2. Compute Hull Area
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    hull_idx = hull.vertices
                except Exception:
                    return -1e10, np.zeros_like(p_flat)
                    
                if hull_area < 1e-10:
                    return -1e10, np.zeros_like(p_flat)
                    
                log_hull_area = np.log(hull_area)
                
                # Gradient of log_hull_area
                grad_log_hull = np.zeros_like(pts)
                
                for m in range(len(hull_idx)):
                    curr = hull_idx[m]
                    prev = hull_idx[(m - 1) % len(hull_idx)]
                    next_ = hull_idx[(m + 1) % len(hull_idx)]
                    
                    dA_dx = 0.5 * (pts[prev, 1] - pts[next_, 1])
                    dA_dy = 0.5 * (pts[next_, 0] - pts[prev, 0])
                    
                    grad_log_hull[curr, 0] += dA_dx / hull_area
                    grad_log_hull[curr, 1] += dA_dy / hull_area
                    
                # Objective: Minimize log_hull_area - soft_min_val
                val = log_hull_area - soft_min_val
                grad = grad_log_hull - grad_soft_min
                
                return val, grad.flatten()

            # Run L-BFGS-B
            try:
                res = minimize(
                    obj_and_grad,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
                )
                current_pts = res.x.reshape(n, 2)
            except Exception:
                pass
            
            # Update best
            score, _ = compute_score(current_pts)
            if score > best_score:
                best_score = score
                best_points = current_pts.copy()
                
                # Local refinement: try small perturbations
                if time.time() - start_time < time_limit - safety_margin - 10:
                    for _ in range(5):
                        perturbed = current_pts + rng.normal(0, 0.01, (n, 2))
                        pert_score, _ = compute_score(perturbed)
                        if pert_score > best_score:
                            best_score = pert_score
                            best_points = perturbed.copy()
    
    if best_points is None:
        return base_poly
        
    return best_points

# EVOLVE-BLOCK-END
