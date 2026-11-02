import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def calc_radius(c):
        # Min pairwise distance / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Min distance to wall
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def init_grid():
        x_coords = np.linspace(-0.4, 0.4, 5)
        y_coords = np.linspace(-0.4, 0.4, 5)
        z_coords = np.linspace(-0.375, 0.375, 4)
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        centers = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return centers[:N]

    def optimize_l_bfgsb(centers):
        def objective_and_grad(flat_centers):
            c = flat_centers.reshape(-1, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
            
            val = np.sum(1.0 / dist)
            
            dist3 = dist ** 3
            weights = 1.0 / dist3
            np.fill_diagonal(weights, 0.0)
            
            term1 = weights @ c
            term2 = weights.sum(axis=1)[:, np.newaxis] * c
            grad = -(term1 - term2)
            
            return val, grad.ravel()

        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective_and_grad, 
                centers.ravel(), 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            return res.x.reshape(-1, 3)
        except Exception:
            return centers

    def local_search(centers, time_limit):
        current_radius = calc_radius(centers)
        best_centers = centers.copy()
        best_radius = current_radius
        
        temp = current_radius * 0.1
        cool_rate = 0.999
        
        no_improve_count = 0
        
        while time.time() < time_limit:
            # Pick a random point
            idx = np.random.randint(N)
            
            # Generate perturbation
            step = temp * 0.5
            delta = np.random.normal(0, step, size=3)
            
            new_center = centers[idx].copy() + delta
            new_center = np.clip(new_center, -0.5, 0.5)
            
            old_center = centers[idx].copy()
            centers[idx] = new_center
            
            # Calculate new radius efficiently
            # Only need to check distances involving idx
            # and the previous global min (if it didn't involve idx)
            # For simplicity and safety, recalc full radius if N is small?
            # N=100, full recalc is 10k ops.
            # Let's try full recalc first, it's simple.
            new_radius = calc_radius(centers)
            
            delta_r = new_radius - current_radius
            
            if new_radius > current_radius or np.random.rand() < np.exp(delta_r / temp):
                centers[idx] = new_center
                current_radius = new_radius
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                centers[idx] = old_center
                no_improve_count += 1
                
            temp *= cool_rate
            
            if no_improve_count > 1000:
                break
                
        return best_centers, best_radius

    # Main Loop
    best_radius = 0
    best_centers = init_grid()
    
    # Phase 1: Grid -> LBFGS -> Local Search
    centers = init_grid()
    centers = optimize_l_bfgsb(centers)
    centers, r = local_search(centers, deadline - 20.0)
    if r > best_radius:
        best_radius = r
        best_centers = centers

    # Phase 2: If time remains, try another restart
    if time.time() < deadline - 30.0:
        # Perturb the best solution
        centers = best_centers + np.random.normal(0, 0.01, size=(N, 3))
        centers = np.clip(centers, -0.5, 0.5)
        
        centers = optimize_l_bfgsb(centers)
        centers, r = local_search(centers, deadline - 10.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers

    # Phase 3: If time remains, try a different grid orientation or random start
    if time.time() < deadline - 30.0:
        # Random Start
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        centers = optimize_l_bfgsb(centers)
        centers, r = local_search(centers, deadline - 10.0)
        if r > best_radius:
            best_radius = r
            best_centers = centers

    return best_centers

# EVOLVE-BLOCK-END
