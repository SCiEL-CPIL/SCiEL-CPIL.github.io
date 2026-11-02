import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Fixed input codes for the MLP
    # We use a deterministic random projection of the indices to break symmetry
    # while providing a stable input.
    rng = np.random.RandomState(42)
    Z = rng.randn(N, 16)
    
    # 2. Grid initialization for the base positions
    # 4x5x5 grid = 100 points
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    
    # 3. MLP Architecture: 16 -> 32 -> 32 -> 3
    # The MLP outputs a deviation from the grid.
    # Parameters:
    # W1: (16, 32), b1: (32,)
    # W2: (32, 32), b2: (32,)
    # W3: (32, 3),  b3: (3,)
    
    # Initialize weights small, biases zero (so initial output is 0, centers = grid)
    scale = 0.1
    W1 = scale * rng.randn(16, 32)
    b1 = np.zeros(32)
    W2 = scale * rng.randn(32, 32)
    b2 = np.zeros(32)
    W3 = scale * rng.randn(32, 3)
    b3 = np.zeros(3)
    
    # Helper to flatten parameters
    def get_params():
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
    
    # Helper to unpack parameters
    def set_params(p):
        nonlocal W1, b1, W2, b2, W3, b3
        # Sizes:
        # W1: 16*32 = 512
        # b1: 32
        # W2: 32*32 = 1024
        # b2: 32
        # W3: 32*3 = 96
        # b3: 3
        
        idx = 0
        W1 = p[idx:idx+512].reshape(16, 32)
        idx += 512
        b1 = p[idx:idx+32]
        idx += 32
        W2 = p[idx:idx+1024].reshape(32, 32)
        idx += 1024
        b2 = p[idx:idx+32]
        idx += 32
        W3 = p[idx:idx+96].reshape(32, 3)
        idx += 96
        b3 = p[idx:idx+3]
        
    def forward(p):
        set_params(p)
        # Layer 1
        h1 = np.tanh(Z @ W1 + b1)
        # Layer 2
        h2 = np.tanh(h1 @ W2 + b2)
        # Layer 3 (Linear output for deviation)
        delta = h2 @ W3 + b3
        centers = grid + delta
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(p, tau):
        centers = forward(p)
        
        # Pairwise distances
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

    # 4. Optimization
    p0 = get_params()
    
    # Multi-stage optimization with decreasing temperature
    taus = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    p_opt = p0
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
        p_opt = res.x
        
    centers = forward(p_opt)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
