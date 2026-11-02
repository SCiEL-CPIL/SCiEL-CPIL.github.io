import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Initial Grid Setup
    # Use a slightly irregular grid to break symmetry and provide a better starting landscape
    x = np.linspace(-0.45, 0.45, 4)
    y = np.linspace(-0.45, 0.45, 5)
    z = np.linspace(-0.45, 0.45, 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])[:N]
    
    # Add small random jitter to break strict grid symmetry
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.005, size=grid.shape)
    
    # 2. Neural Network Parameterization
    # We define a small MLP that maps an index vector to a displacement vector.
    # Input: Index i (normalized to [-1, 1])
    # Hidden Layers: 2 layers of size 24 with tanh activation
    # Output: 3 dimensions (x, y, z displacement)
    
    # Precompute input features for all N points
    idx = np.linspace(-1, 1, N).reshape(-1, 1)
    
    # Initialize weights to small values so the network starts near zero displacement
    # This ensures the initial state is close to the valid grid.
    W1 = rng.normal(0, 0.1, size=(1, 24))
    b1 = np.zeros(24)
    W2 = rng.normal(0, 0.1, size=(24, 24))
    b2 = np.zeros(24)
    W3 = rng.normal(0, 0.1, size=(24, 3))
    b3 = np.zeros(3)
    
    # Flatten parameters for optimization
    p0 = np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
    total_params = len(p0)
    
    def decode(p):
        W1 = p[0:24].reshape(1, 24)
        b1 = p[24:48]
        W2 = p[48:48+24*24].reshape(24, 24)
        b2 = p[48+24*24:48+24*24+24]
        W3 = p[48+24*24+24:48+24*24+24+24*3].reshape(24, 3)
        b3 = p[48+24*24+24+24*3:]
        
        # Forward pass
        h1 = np.tanh(idx @ W1.T + b1)
        h2 = np.tanh(h1 @ W2.T + b2)
        disp = h2 @ W3.T + b3
        
        centers = grid + disp
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective_nn(p, tau):
        centers = decode(p)
        
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

    # 3. Optimization Stage 1: Optimize Neural Network Parameters
    # Use a schedule of decreasing temperatures to refine the packing
    tau_schedule = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    p_opt = p0.copy()
    for tau in tau_schedule:
        res = minimize(
            lambda p: objective_nn(p, tau), 
            p_opt, 
            method='L-BFGS-B', 
            options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100}
        )
        p_opt = res.x
        
    # Get centers from the optimized network
    centers_nn = decode(p_opt)

    # 4. Optimization Stage 2: Direct Coordinate Polish
    # The neural network might get stuck in a local minimum of the latent space.
    # We now optimize the 300 coordinates directly. This is high-dimensional 
    # but starting from the NN result often finds a better local optimum 
    # than starting from the raw grid.
    
    def objective_direct(c_flat, tau):
        centers = c_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # Start with a slightly higher tau for robustness, then lower it
    tau_direct = [0.01, 0.005, 0.002, 0.001, 0.0005]
    c_opt = centers_nn.ravel()
    
    for tau in tau_direct:
        res = minimize(
            lambda c: objective_direct(c, tau), 
            c_opt, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        c_opt = res.x
        
    centers_final = c_opt.reshape(N, 3)
    
    # Final safety clip to ensure no point is exactly on the boundary or outside
    # The clip value is very close to 0.5 to avoid violating constraints while maximizing radius
    centers_final = np.clip(centers_final, -0.499999, 0.499999)
    
    return centers_final
