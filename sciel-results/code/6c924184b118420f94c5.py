import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define Latent Codes for each sphere
    # We use a deterministic set of latent vectors.
    # A common strategy is to use the indices of a grid or random fixed vectors.
    # Here, we use a mix: initialize centers on a grid, then use their indices 
    # or normalized coordinates as input to the network? 
    # Actually, a simpler and effective "neural" approach for packing is to 
    # parameterize the displacement of a base grid using a small MLP that takes
    # the grid index as input.
    
    # Let's create a base grid. 5x4x5 = 100 points.
    x_coords = np.linspace(-0.45, 0.45, 5)
    y_coords = np.linspace(-0.45, 0.45, 4)
    z_coords = np.linspace(-0.45, 0.45, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    base_centers = grid[:N]
    
    # 2. Define the Neural Network (MLP)
    # Input: Index of the sphere (0 to 99) normalized to [0,1] or just the base center?
    # Using the base center as input is better because it provides spatial context.
    # Input dim: 3
    # Hidden layer 1: 16 units
    # Hidden layer 2: 16 units
    # Output dim: 3 (displacement)
    
    # Weights and Biases
    # W1: (3, 16), b1: (16,)
    # W2: (16, 16), b2: (16,)
    # W3: (16, 3), b3: (3,)
    
    # Initialize weights with small random values (deterministic seed)
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 0.1, (3, 16))
    b1 = rng.normal(0, 0.1, 16)
    W2 = rng.normal(0, 0.1, (16, 16))
    b2 = rng.normal(0, 0.1, 16)
    W3 = rng.normal(0, 0.01, (16, 3)) # Small output to start close to base
    b3 = rng.normal(0, 0.01, 3)
    
    # Flatten parameters
    params0 = np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
    
    def mlp_forward(params, inputs):
        W1 = params[0:48].reshape(3, 16)
        b1 = params[48:64]
        W2 = params[64:112].reshape(16, 16)
        b2 = params[112:128]
        W3 = params[128:176].reshape(16, 3)
        b3 = params[176:179]
        
        h1 = np.tanh(inputs @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        disp = h2 @ W3 + b3
        return disp

    def centers_from_params(params):
        disp = mlp_forward(params, base_centers)
        centers = base_centers + disp
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
        
        # Maximize the minimum constraint
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization of MLP Parameters
    # Multi-stage annealing
    taus = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    params_opt = params0
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), params_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        params_opt = res.x
        # Optional: Restart if stuck? For simplicity, we just continue.
        
    centers = centers_from_params(params_opt)

    # 4. Direct Coordinate Polish
    # The MLP provides a good structured initialization. Now refine directly.
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

    taus_direct = [0.005, 0.001, 0.0005, 0.0001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers
