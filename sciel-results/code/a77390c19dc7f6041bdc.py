import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Base Grid
    # 4x5x5 grid
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Neural Network Parameterization
    # Latent dimension
    D = 64
    # Hidden layers
    H1 = 128
    H2 = 128
    
    rng = np.random.default_rng(42)
    
    # Initialize weights with small values to start near the grid
    # He initialization is common, but we want small displacements initially.
    # So we scale down.
    W1 = rng.normal(0, 1.0 / np.sqrt(D), size=(H1, D))
    b1 = np.zeros(H1)
    W2 = rng.normal(0, 1.0 / np.sqrt(H1), size=(H2, H1))
    b2 = np.zeros(H2)
    W3 = rng.normal(0, 1.0 / np.sqrt(H2), size=(300, H2))
    b3 = np.zeros(300)
    
    # Scale factor for displacement
    disp_scale = 0.25 # Max displacement ~0.25, enough to move from grid to optimal
    
    def centers_from_z(z):
        h1 = np.tanh(W1 @ z + b1)
        h2 = np.tanh(W2 @ h1 + b2)
        disp = W3 @ h2 + b3
        disp = np.tanh(disp) * disp_scale
        centers = grid + disp.reshape(N, 3)
        return centers

    # 3. Objective Function
    # Precompute indices for efficiency
    iu = np.triu_indices(N, k=1)
    
    def softmin(x, tau=0.01):
        # Numerically stable softmin
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(z):
        centers = centers_from_z(z)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        
        # We want to maximize the minimum constraint.
        # softmin approximates min.
        # Using a smaller tau makes it closer to min but sharper gradients.
        sm = softmin(all_constraints, tau=0.005)
        return -sm

    # 4. Optimization Loop
    best_z = np.zeros(D)
    best_val = -np.inf
    
    # Multiple restarts
    for trial in range(5):
        if trial == 0:
            z0 = np.zeros(D)
        else:
            z0 = rng.normal(0, 0.5, size=D)
            
        res = minimize(objective, z0, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        val = -res.fun
        if val > best_val:
            best_val = val
            best_z = res.x

    centers = centers_from_z(best_z)
    
    # 5. Local Polish (Direct Coordinate Optimization)
    # This step refines the centers directly, allowing for more precise adjustments
    # that the latent space might have smoothed over.
    
    def objective_direct(centers_flat):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau=0.001)
        return -sm

    # Run polish from the best latent solution
    res2 = minimize(objective_direct, centers.ravel(), method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    centers = res2.x.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers
