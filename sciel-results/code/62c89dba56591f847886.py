"""Improved packing of 50 equal spheres in a unit cube using smooth optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit

    def score(c):
        """Compute the minimum radius for given centers."""
        c = c.reshape(N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective(x, beta):
        """Smooth surrogate for the min-radius objective using log-sum-exp."""
        c = x.reshape(N, 3)
        # Pairwise squared distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        # Wall squared distances
        wall_d2 = np.minimum(c + 0.5, 0.5 - c) ** 2
        wall_d2 = np.min(wall_d2, axis=1)
        
        # Combine all constraints
        # We want to maximize the minimum of (d/2) and (wall_dist)
        # Equivalently, minimize the maximum of (2/d) and (1/wall_dist)
        # Or use a smooth min via log-sum-exp
        
        # Use log-sum-exp on the negative of the log of distances
        # For pairwise: we want min of d/2
        # For walls: we want min of wall_dist
        
        # Let's use a different approach: smooth min via -log(sum(exp(-beta * x)))
        # This approximates min(x) as beta -> inf
        
        # Pairwise terms
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        pairwise_terms = d / 2.0
        
        # Wall terms
        wall_dist = np.sqrt(wall_d2)
        
        # Combine all terms
        all_terms = np.concatenate([pairwise_terms[np.triu_indices(N, k=1)], wall_dist])
        
        # Smooth min: -1/beta * log(sum(exp(-beta * terms)))
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        smooth_min = max_term - np.log(np.sum(exp_terms)) / beta
        
        return -smooth_min

    def smooth_gradient(x, beta):
        """Gradient of the smooth objective."""
        c = x.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        
        wall_d2 = np.minimum(c + 0.5, 0.5 - c) ** 2
        wall_d2 = np.min(wall_d2, axis=1)
        wall_dist = np.sqrt(wall_d2)
        
        pairwise_terms = d / 2.0
        all_terms = np.concatenate([pairwise_terms[np.triu_indices(N, k=1)], wall_dist])
        
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        weights = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(x)
        # Pairwise gradient
        i_idx, j_idx = np.triu_indices(N, k=1)
        for idx in range(len(i_idx)):
            i, j = i_idx[idx], j_idx[idx]
            w = weights[idx]
            vec = c[i] - c[j]
            dist = d[i, j]
            if dist > 1e-12:
                grad[i*3:i*3+3] += w * vec / (2 * dist)
                grad[j*3:j*3+3] -= w * vec / (2 * dist)
        
        # Wall gradient
        wall_start = len(i_idx)
        for idx in range(N):
            w = weights[wall_start + idx]
            for k in range(3):
                c_val = c[idx, k]
                if c_val > 0:
                    # Closest wall is +0.5
                    grad[idx*3+k] -= w
                else:
                    # Closest wall is -0.5
                    grad[idx*3+k] += w
        
        return -grad  # Negative because we're minimizing -smooth_min

    # Initial guess: Use a BCC-like lattice
    m = 3
    corner_points = []
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(m + 1):
                corner_points.append([i, j, k])
    
    body_center_points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                body_center_points.append([i + 0.5, j + 0.5, k + 0.5])
    
    all_points = np.array(corner_points + body_center_points)
    all_points = all_points / m - 0.5
    
    np.random.seed(42)
    indices = np.random.choice(len(all_points), N, replace=False)
    centers = all_points[indices].copy()
    
    best_centers = centers.copy()
    best_r = score(centers)
    
    # Phase 1: Smooth optimization with increasing beta
    betas = [50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 30:
            break
        
        x0 = best_centers.flatten()
        
        def obj(x):
            return smooth_objective(x, beta)
        
        def grad(x):
            return smooth_gradient(x, beta)
        
        try:
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            new_centers = result.x.reshape(N, 3)
            new_r = score(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    # Phase 2: Direct optimization on the raw min objective
    if time.time() - start_time < time_limit - 15:
        def raw_objective(x):
            c = x.reshape(N, 3)
            return -score(c)
        
        def raw_gradient(x):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            r = min(np.min(d) / 2.0, np.min(dist_to_walls))
            
            G = np.zeros_like(c)
            eps = 1e-6
            
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 2 * r + eps:
                        vec = c[i] - c[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            G[i] += vec / norm
                            G[j] -= vec / norm
            
            for i in range(N):
                if dist_to_walls[i] < r + eps:
                    for k in range(3):
                        if c[i, k] > 0:
                            G[i, k] -= 1.0
                        else:
                            G[i, k] += 1.0
            
            return -G.flatten()
        
        try:
            result = minimize(
                raw_objective,
                best_centers.flatten(),
                jac=raw_gradient,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            new_centers = result.x.reshape(N, 3)
            new_r = score(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    # Phase 3: Final polish with small perturbations
    if time.time() - start_time < time_limit - 5:
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            centers = best_centers.copy()
            np.random.seed(int(time.time() * 1000) % 100000)
            centers += np.random.normal(0, 0.001, centers.shape)
            centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Quick local optimization
            try:
                result = minimize(
                    raw_objective,
                    centers.flatten(),
                    jac=raw_gradient,
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_r = score(new_centers)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    
            except Exception:
                pass
    
    return best_centers
