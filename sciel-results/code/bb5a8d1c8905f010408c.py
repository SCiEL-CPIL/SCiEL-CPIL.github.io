import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Initialization: 5x5x4 grid
    # This provides a dense, symmetric starting point.
    x_coords = np.linspace(-0.4, 0.4, 5)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define a small MLP to generate displacements
    # Input: grid coordinates (N, 3)
    # Hidden: tanh(W1 @ X + b1)
    # Output: W2 @ H + b2
    hidden_dim = 32
    
    np.random.seed(42)
    # Initialize weights with small random values
    W1 = np.random.randn(3, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, 3) * 0.1
    b2 = np.zeros(3)
    
    def flatten_params(params):
        W1 = params[:3*hidden_dim].reshape(3, hidden_dim)
        b1 = params[3*hidden_dim:4*hidden_dim]
        W2 = params[4*hidden_dim:4*hidden_dim + hidden_dim*3].reshape(hidden_dim, 3)
        b2 = params[4*hidden_dim + hidden_dim*3:]
        return W1, b1, W2, b2

    def centers_from_params(params):
        W1, b1, W2, b2 = flatten_params(params)
        # Forward pass
        H = np.tanh(grid @ W1 + b1)
        disp = H @ W2 + b2
        centers = grid + disp
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(params, tau):
        centers = centers_from_params(params)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # Initial parameters
    W1_init = W1.ravel()
    b1_init = b1.ravel()
    W2_init = W2.ravel()
    b2_init = b2.ravel()
    params_init = np.concatenate([W1_init, b1_init, W2_init, b2_init])
    
    # 3. Optimization Stage 1: MLP Parameters
    # Coarse to fine tau
    taus_mlp = [0.05, 0.02, 0.01, 0.005, 0.002]
    p_opt = params_init
    for tau in taus_mlp:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        p_opt = res.x
        
    centers = centers_from_params(p_opt)
    
    # 4. Optimization Stage 2: Direct Coordinates Refinement
    # The MLP might get stuck in local minima or not fully exploit the space.
    # Refine with direct coordinates using the MLP output as a good start.
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

    taus_direct = [0.002, 0.001, 0.0005]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
        
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
