import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # We will try several initialization strategies and optimize each.
    # The optimization minimizes an energy function that penalizes close spheres and wall violations.
    
    def energy_and_grad(x):
        """
        x: (N*3,) array of centers.
        Returns energy and gradient.
        Energy = sum_{i<j} 1/d^2
        """
        centers = x.reshape(N, 3)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1.0) # Avoid division by zero
        
        # E = sum_{i<j} 1/d^2 = 0.5 * sum_{i!=j} 1/d^2
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        E = 0.5 * np.sum(inv_d2)
        
        # Gradient
        # dE/dc_i = - sum_{j!=i} (c_i - c_j) / d_{ij}^4
        d4 = d2**2
        np.fill_diagonal(d4, 1.0)
        # grad_i = - sum_j diff[i,j] / d4[i,j]
        grad = -1.0 * np.sum(diff / d4[:, :, np.newaxis], axis=1)
        
        return E, grad.flatten()

    def optimize(config):
        x0 = config.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        res = minimize(
            fun=lambda x: energy_and_grad(x)[0],
            x0=x0,
            jac=lambda x: energy_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_centers = res.x.reshape(N, 3)
        
        # Calculate radius
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1.0)
        d_min = np.sqrt(np.min(d2[np.triu_indices(N, k=1)]))
        w_min = np.min(0.5 - np.abs(best_centers))
        r = min(d_min / 2.0, w_min)
        
        return best_centers, r

    def generate_initializations():
        inits = []
        
        # 1. Jittered Grid
        m = 4 # 4x4x4 = 64
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = np.random.normal(0, 0.01, grid.shape)
        grid_jittered = np.clip(grid + jitter, -0.49, 0.49)
        inits.append(grid_jittered)
        
        # 2. Random
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
        # 3. Spectral (Eigenvectors of random SPD matrix)
        # Generate random symmetric matrix
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        # Make it positive definite
        A = A @ A.T + np.eye(N) * 0.1
        # Eigenvectors
        eigvecs = np.linalg.eig(A)[1]
        # Take first N? We have N eigenvectors.
        # Project to 3D? The directive says "representing sphere centers as top N eigenvectors".
        # This implies the coordinates are the components of the eigenvectors?
        # If we take the first 3 components of each eigenvector, they are in R^3.
        # Let's normalize and scale them to fit in the cube.
        spectral_centers = eigvecs[:, :3]
        # Center and scale
        spectral_centers -= np.mean(spectral_centers, axis=0)
        spectral_centers /= np.max(np.abs(spectral_centers))
        spectral_centers *= 0.4 # Fit in [-0.4, 0.4]
        inits.append(spectral_centers)
        
        # 4. Another random
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
        return inits

    inits = generate_initializations()
    
    best_overall_r = 0.0
    best_overall_centers = None
    
    # Time management
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave buffer for 120s
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        centers, r = optimize(init)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers
            
    # If we have time, do a final polish on the best one
    if time.time() - start_time < time_limit and best_overall_centers is not None:
        # Add small noise to escape local min
        noisy = best_overall_centers + np.random.normal(0, 0.005, best_overall_centers.shape)
        noisy = np.clip(noisy, -0.49, 0.49)
        centers, r = optimize(noisy)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers
            
    return best_overall_centers
