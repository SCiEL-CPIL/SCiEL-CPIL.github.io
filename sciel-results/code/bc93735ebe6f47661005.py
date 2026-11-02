import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # A 4x5x5 grid provides 100 points
    x_coords = np.linspace(-0.45, 0.45, 4)
    y_coords = np.linspace(-0.45, 0.45, 5)
    z_coords = np.linspace(-0.45, 0.45, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Fourier Feature Generation
    # We use Fourier features of the initial grid positions as input to the MLP.
    # This allows the network to learn a non-linear displacement field.
    
    K = 4 # Max frequency
    # Generate frequency vectors k = (kx, ky, kz) where 1 <= kx,ky,kz <= K
    kx = np.arange(1, K+1)
    ky = np.arange(1, K+1)
    kz = np.arange(1, K+1)
    
    # Create meshgrid for frequencies
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    freqs = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1) # Shape (K^3, 3)
    
    # Compute dot products: grid (N, 3) @ freqs.T (3, K^3) -> (N, K^3)
    proj = grid @ freqs.T
    proj = np.pi * proj # Scale by pi to map [-0.5, 0.5] to [-pi/2, pi/2] roughly, or just use pi * k * x
    
    # Features: cos and sin
    phi = np.hstack([np.cos(proj), np.sin(proj)]) # Shape (N, 2*K^3)
    # Add bias term
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_features = phi.shape[1] # 1 + 2*64 = 129
    
    # 3. Define MLP Architecture
    # Input: n_features
    # Hidden: 64 units, ReLU
    # Output: 3 units (displacement)
    
    hidden_dim = 64
    
    # Initialize weights
    # Using small random initialization to break symmetry
    np.random.seed(42)
    W1 = np.random.randn(n_features, hidden_dim) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, 3) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(3)
    
    # Flatten parameters for L-BFGS-B
    params = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
    
    def unpack_params(p):
        W1_flat = p[:n_features * hidden_dim]
        b1_vec = p[n_features * hidden_dim : n_features * hidden_dim + hidden_dim]
        W2_flat = p[n_features * hidden_dim + hidden_dim : n_features * hidden_dim + hidden_dim + hidden_dim * 3]
        b2_vec = p[n_features * hidden_dim + hidden_dim + hidden_dim * 3 :]
        
        W1_mat = W1_flat.reshape(n_features, hidden_dim)
        W2_mat = W2_flat.reshape(hidden_dim, 3)
        
        return W1_mat, b1_vec, W2_mat, b2_vec
    
    def get_centers(p):
        W1_mat, b1_vec, W2_mat, b2_vec = unpack_params(p)
        # Forward pass
        h = phi @ W1_mat + b1_vec # (N, hidden_dim)
        h = np.maximum(h, 0) # ReLU
        disp = h @ W2_mat + b2_vec # (N, 3)
        centers = grid + disp
        return centers
    
    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))
    
    def objective(p, tau):
        centers = get_centers(p)
        
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
    # Multi-stage optimization with decreasing tau
    taus = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    p_opt = params
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
        p_opt = res.x
        
    centers = get_centers(p_opt)
    
    # 5. Local Polish with Direct Coordinates
    # The MLP might be slightly constrained. We can refine the centers directly.
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

    taus_direct = [0.001, 0.0005]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
