import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # 4 x 5 x 5 grid
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define the Neural Network with Fourier Features
    # Latent dimension D
    D = 20
    # Fourier feature dimension
    F = 64
    # Hidden layer dimension
    H = 128
    
    rng = np.random.default_rng(42)
    
    # Fourier Feature Map: z (D,) -> (F,)
    # Random frequencies
    B = rng.normal(0, 1.0, size=(D, F // 2))
    
    def fourier_features(z):
        # z: (D,)
        # cos(B @ z) and sin(B @ z)
        proj = z @ B
        return np.concatenate([np.cos(proj), np.sin(proj)])
        
    # MLP: F -> H -> 300
    W1 = rng.normal(0, 1.0 / np.sqrt(F), size=(H, F))
    b1 = rng.normal(0, 0.1, size=(H,))
    W2 = rng.normal(0, 1.0 / np.sqrt(H), size=(300, H))
    b2 = rng.normal(0, 0.1, size=(300,))
    
    # Scale factor for displacement to ensure it starts small relative to grid spacing
    # Grid spacing is 0.8/3 = 0.266. Min distance is 0.266.
    # We want to allow movement but not break the grid initially.
    scale = 0.15 
    
    def centers_from_z(z):
        h_feat = fourier_features(z)
        h = np.tanh(W1 @ h_feat + b1)
        disp = W2 @ h + b2
        disp = np.tanh(disp) * scale # Keep displacement bounded
        centers = grid + disp.reshape(N, 3)
        return centers

    # 3. Objective Function
    def softmin(x, tau=0.01):
        m = np.min(x)
        # Numerical stability
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(z, tau):
        centers = centers_from_z(z)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        
        sm = softmin(all_constraints, tau=tau)
        return -sm

    # 4. Optimization
    best_z = np.zeros(D)
    best_val = -np.inf
    
    # Multi-stage optimization with decreasing temperature
    taus = [0.05, 0.02, 0.005, 0.001]
    
    for tau in taus:
        for trial in range(2):
            if trial == 0:
                z0 = np.zeros(D) if best_val == -np.inf else best_z + rng.normal(0, 0.01, size=D)
            else:
                z0 = rng.normal(0, 0.1, size=D)
                
            res = minimize(lambda z: objective(z, tau), z0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000})
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_z = res.x

    centers = centers_from_z(best_z)
    
    # 5. Local Polish
    # Optimize the 300 coordinates directly to fine-tune
    def objective_direct(centers_flat, tau=0.001):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau=tau)
        return -sm

    # Use a very small tau for the direct polish to get close to the true min
    res2 = minimize(lambda c: objective_direct(c, tau=0.001), centers.ravel(), method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000})
    centers = res2.x.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
