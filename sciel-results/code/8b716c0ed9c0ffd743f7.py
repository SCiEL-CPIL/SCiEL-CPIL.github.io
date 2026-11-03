import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Energy function for L-BFGS-B
    # We minimize E = sum 1/d^2 + sum 1/w^2
    # This encourages large distances and large wall clearance.
    
    def energy_and_grad(x):
        centers = x.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        # Avoid division by zero
        d2 = np.maximum(d2, 1e-12)
        d = np.sqrt(d2)
        d3 = d ** 3
        
        # Gradient of sum 1/d^2
        # d/dx_i (1/d_ij^2) = -2 (x_i - x_j) / d_ij^4
        # Sum over j
        # force_spheres[i] = sum_j -2 (c_i - c_j) / d_ij^4
        # Note: diff[i,j] = c_i - c_j
        # term = diff / d4
        d4 = d ** 4
        np.fill_diagonal(d4, 1e-12)
        
        force_spheres = np.sum(-2.0 * diff / d4[:, :, np.newaxis], axis=1)
        
        # Wall distances
        # w_ik = 0.5 - |c_ik|
        # Penalty 1/w^2
        # d/dc (1/(0.5-c)^2) = 2/(0.5-c)^3
        # d/dc (1/(0.5+c)^2) = -2/(0.5+c)^3
        
        force_walls = np.zeros_like(centers)
        for k in range(3):
            c = centers[:, k]
            dist_pos = 0.5 - c
            dist_neg = c + 0.5
            
            # Clip to avoid singularity
            dist_pos = np.maximum(dist_pos, 1e-6)
            dist_neg = np.maximum(dist_neg, 1e-6)
            
            # Gradient wrt c_k
            # If c > 0, w = 0.5 - c. d(1/w^2)/dc = 2/w^3
            # If c < 0, w = 0.5 + c. d(1/w^2)/dc = -2/w^3
            # We can combine:
            # f_pos = 2 / dist_pos^3
            # f_neg = -2 / dist_neg^3
            # Total force = f_pos * (c>0) + f_neg * (c<0) ?
            # Actually, the function 1/(0.5-|c|)^2 is not differentiable at 0.
            # But for optimization, we can just use the subgradient or smooth it.
            # Let's use a smooth approximation: 1/(0.5 - c)^2 + 1/(0.5 + c)^2 ? No, that penalizes both sides.
            # The actual constraint is |c| <= 0.5 - r.
            # Let's just penalize the distance to the nearest wall.
            # w = 0.5 - np.abs(c)
            # E_wall = 1/w^2
            # dE/dc = -2/w^3 * d(w)/dc = -2/w^3 * (-sign(c)) = 2*sign(c)/w^3
            
            sign_c = np.sign(c)
            w = 0.5 - np.abs(c)
            w = np.maximum(w, 1e-6)
            
            grad_wall = 2.0 * sign_c / (w ** 3)
            
            force_walls[:, k] = grad_wall
        
        total_grad = force_spheres + force_walls
        
        # Energy value
        E_spheres = np.sum(1.0 / d2[np.triu_indices(N, k=1)])
        E_walls = np.sum(1.0 / (0.5 - np.abs(centers))**2)
        # Clip to avoid inf
        E = np.min(E_spheres + E_walls, 1e6)
        
        return E, total_grad.flatten()

    def get_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        d_min = np.min(np.sqrt(d2[np.triu_indices(N, k=1)]))
        w_min = np.min(0.5 - np.abs(centers))
        return min(d_min / 2.0, w_min)

    # Initialization Strategies
    def init_spectral():
        # Random SPD matrix
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        A += np.eye(N) * 0.5 # Ensure PD
        eigvecs = np.linalg.eig(A)[1]
        # Take first 3 components of each eigenvector
        centers = eigvecs[:, :3]
        # Normalize and scale
        centers -= np.mean(centers, axis=0)
        max_val = np.max(np.abs(centers))
        if max_val > 0:
            centers /= max_val
        centers *= 0.45 # Fit in cube
        return centers.flatten()

    def init_grid():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # We need 50. 4x4x4=64.
        # Shuffle and take 50
        np.random.shuffle(grid)
        grid = grid[:N]
        # Jitter
        grid += np.random.normal(0, 0.02, grid.shape)
        grid = np.clip(grid, -0.45, 0.45)
        return grid.flatten()

    def init_random():
        return np.random.uniform(-0.4, 0.4, (N, 3)).flatten()

    inits = [
        init_spectral(),
        init_grid(),
        init_random(),
        init_grid() # Another grid
    ]

    best_r = 0.0
    best_centers = None
    start_time = time.time()
    time_limit = 100.0

    for i, x0 in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        try:
            # Stage 1: Coarse optimization
            res = minimize(
                energy_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            centers = res.x.reshape(N, 3)
            r = get_radius(centers)
            
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
                
            # Stage 2: Fine optimization on the best so far if time permits
            if time.time() - start_time < time_limit - 10:
                res2 = minimize(
                    energy_and_grad, 
                    centers.flatten(), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                centers2 = res2.x.reshape(N, 3)
                r2 = get_radius(centers2)
                if r2 > best_r:
                    best_r = r2
                    best_centers = centers2.copy()
                    
        except Exception as e:
            pass

    if best_centers is None:
        # Fallback
        best_centers = init_grid().reshape(N, 3)

    return best_centers
