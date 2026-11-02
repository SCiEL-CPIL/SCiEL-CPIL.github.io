import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # A 4x5x5 grid provides 100 points exactly
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define a small MLP to generate displacements
    # Input: Grid coordinates (N, 3)
    # Hidden Layer 1: 16 neurons, tanh activation
    # Hidden Layer 2: 8 neurons, tanh activation
    # Output: 3 (displacements)
    
    # Initialize weights with small random values for symmetry breaking
    # We use a fixed seed for reproducibility if needed, but random is fine for search
    np.random.seed(42)
    
    h1_size = 16
    h2_size = 8
    
    # Weights: (in, out)
    W1 = np.random.randn(3, h1_size) * 0.1
    b1 = np.zeros(h1_size)
    W2 = np.random.randn(h1_size, h2_size) * 0.1
    b2 = np.zeros(h2_size)
    W3 = np.random.randn(h2_size, 3) * 0.1
    b3 = np.zeros(3)
    
    # Flatten parameters
    param_vec = np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
    
    def mlp_forward(p):
        W1 = p[0:3*h1_size].reshape(3, h1_size)
        b1 = p[3*h1_size:3*h1_size+h1_size]
        W2 = p[3*h1_size+h1_size:3*h1_size+h1_size+h1_size*h2_size].reshape(h1_size, h2_size)
        b2 = p[3*h1_size+h1_size+h1_size*h2_size:3*h1_size+h1_size+h1_size*h2_size+h2_size]
        W3 = p[3*h1_size+h1_size+h1_size*h2_size+h2_size:3*h1_size+h1_size+h1_size*h2_size+h2_size+3*h2_size].reshape(h2_size, 3)
        b3 = p[3*h1_size+h1_size+h1_size*h2_size+h2_size+3*h2_size:]
        
        z1 = grid @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = np.tanh(z2)
        disp = a2 @ W3 + b3
        
        centers = grid + disp
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(p, tau):
        centers = mlp_forward(p)
        
        # Pairwise distances
        # Use broadcasting for memory efficiency if N is small, but N=100 is fine
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        
        # Maximize the minimum constraint
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization
    # Multi-stage optimization with decreasing tau
    # Start with larger tau to explore, then refine
    taus = [0.05, 0.01, 0.001, 0.0001]
    
    p_opt = param_vec.copy()
    
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        p_opt = res.x
        
    centers = mlp_forward(p_opt)
    
    # 4. Stage 2: Direct coordinate optimization (Local Polish)
    # The MLP might not capture the exact optimal geometry due to approximation error.
    # We use the MLP output as an initial guess for direct coordinate optimization.
    
    def objective_direct(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # Further refine with direct coordinates
    taus_direct = [0.001, 0.0001, 0.00001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
