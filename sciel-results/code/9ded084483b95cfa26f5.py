import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Initialize with Hexagonal Lattice
    def generate_hex_grid(N_target, r_init):
        spacing = 2.0 * r_init
        R_est = spacing * np.sqrt(N_target * np.sqrt(3.0) / (2.0 * np.pi))
        R_est *= 1.1
        
        k = int(np.ceil(R_est / spacing)) + 2
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > R_est:
                continue
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if np.hypot(x, y) <= R_est:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((N_target, 2))
            
        if len(points) > N_target:
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)
            points = points[indices[:N_target]]
        elif len(points) < N_target:
            while len(points) < N_target:
                angle = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, 1)
                points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
        
        return points

    r_guess = 0.05
    points = generate_hex_grid(N, r_guess)
    
    max_rad = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_rad > 1.0:
        scale = 0.99 / max_rad
        points *= scale

    best_points = points.copy()
    best_score = -np.inf

    def compute_score(p):
        """Compute the actual packing score."""
        p = p.reshape(-1, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
        min_boundary = np.min(boundary_dists)
        return min(min_pair_dist, min_boundary)

    # Update best score from initial
    best_score = compute_score(best_points)

    # Precompute pair indices for efficiency
    i_idx, j_idx = np.triu_indices(N, 1)
    n_pairs = len(i_idx)

    def make_objective(beta):
        def objective(x):
            p = x.reshape(-1, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            valid_dists = dists[i_idx, j_idx] / 2.0
            boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
            all_vals = np.concatenate([valid_dists, boundary_dists])
            
            m = np.min(all_vals)
            if m < 0:
                return -m + 0.01 * m  # Penalize violations
            exp_vals = np.exp(-beta * (all_vals - m))
            soft_min = m - (1.0/beta) * np.log(np.sum(exp_vals))
            return -soft_min
        return objective

    def make_gradient(beta):
        def objective_grad(x):
            p = x.reshape(-1, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            valid_dists = dists[i_idx, j_idx] / 2.0
            boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
            all_vals = np.concatenate([valid_dists, boundary_dists])
            
            m = np.min(all_vals)
            exp_vals = np.exp(-beta * (all_vals - m))
            S = np.sum(exp_vals)
            weights = exp_vals / S
            
            w_dists = weights[:n_pairs]
            w_bounds = weights[n_pairs:]
            
            grad = np.zeros_like(x)
            
            # Boundary gradient: d(1 - ||p_i||)/dp_i = -p_i / ||p_i||
            norms = np.linalg.norm(p, axis=1)
            norms[norms == 0] = 1.0
            grad_bounds = -p / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
            grad += grad_bounds
            
            # Distance gradient: d(||p_i - p_j||/2)/dp_i = (p_i - p_j) / (2 * ||p_i - p_j||)
            # For each pair (i,j), contribution to grad[i] is w_ij * (p_i - p_j) / (2 * dist_ij)
            # and to grad[j] is w_ij * (p_j - p_i) / (2 * dist_ij)
            
            # Vectorized approach using full weight matrix
            W_full = np.zeros((N, N))
            W_full[i_idx, j_idx] = w_dists
            W_full[j_idx, i_idx] = w_dists
            
            inv_dists = 1.0 / (2.0 * dists)
            np.fill_diagonal(inv_dists, 0.0)
            
            coeff = W_full * inv_dists
            grad_dists = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            grad += grad_dists
            
            return grad
        return objective_grad

    # Optimization with beta schedule
    beta_schedule = [(20.0, 100), (50.0, 80), (100.0, 60), (200.0, 40), (400.0, 30)]
    
    bounds = [(-1.1, 1.1)] * (2 * N)
    
    for beta, max_iter in beta_schedule:
        if time.time() > deadline - 15:
            break
            
        obj_func = make_objective(beta)
        grad_func = make_gradient(beta)
        
        try:
            res = minimize(obj_func, best_points.flatten(), jac=grad_func, 
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            
            if res.x is not None:
                candidate = res.x.reshape(-1, 2)
                # Check validity
                radii = np.hypot(candidate[:, 0], candidate[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                    score = compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        
        except Exception as e:
            pass

    # Final polish with very high beta and tight tolerances
    if time.time() < deadline - 10:
        beta_final = 1000.0
        obj_final = make_objective(beta_final)
        grad_final = make_gradient(beta_final)
        
        try:
            res_final = minimize(obj_final, best_points.flatten(), jac=grad_final,
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False})
            
            if res_final.x is not None:
                candidate = res_final.x.reshape(-1, 2)
                radii = np.hypot(candidate[:, 0], candidate[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                    score = compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
                        
        except Exception as e:
            pass

    # Final validity check
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0):
        max_r = np.max(radii)
        if max_r > 0:
            scale = 0.9999 / max_r
            best_points *= scale

    return best_points
