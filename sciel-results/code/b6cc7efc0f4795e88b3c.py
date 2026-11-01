import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid: 5 x 5 x 4 = 100
    # Spread centers to maximize initial minimum distance
    x_coords = np.linspace(-0.4, 0.4, 5)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Fourier Feature Neural Network
    # Latent z (D=32) -> Fourier features -> MLP -> displacement
    D = 32
    H1 = 64
    H2 = 64
    rng = np.random.default_rng(42)
    
    # Fourier feature matrix: maps z to sin/cos of random frequencies
    freqs = rng.normal(0, 1.0, size=(D, D))
    
    # MLP weights
    W1 = rng.normal(0, 1.0 / np.sqrt(D * 2), size=(H1, D * 2))
    b1 = np.zeros(H1)
    W2 = rng.normal(0, 1.0 / np.sqrt(H1), size=(H2, H1))
    b2 = np.zeros(H2)
    W3 = rng.normal(0, 0.01, size=(N * 3, H2))
    b3 = np.zeros(N * 3)
    
    def centers_from_z(z):
        # Fourier features
        zf = np.concatenate([np.sin(z @ freqs), np.cos(z @ freqs)], axis=1)
        # MLP
        h1 = np.tanh(W1 @ zf + b1)
        h2 = np.tanh(W2 @ h1 + b2)
        disp = W3 @ h2 + b3
        disp = np.tanh(disp) * 0.15  # Bounded displacement
        centers = grid + disp.reshape(N, 3)
        return centers

    # 3. Objective with softmin
    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(z, tau=0.01):
        centers = centers_from_z(z)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # 4. Multi-stage optimization
    best_z = np.zeros(D)
    best_val = -np.inf
    
    # Stage 1: Coarse optimization
    z0 = np.zeros(D)
    res1 = minimize(lambda z: objective(z, tau=0.05), z0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    
    val1 = -res1.fun
    if val1 > best_val:
        best_val = val1
        best_z = res1.x
    
    # Stage 2: Fine optimization
    res2 = minimize(lambda z: objective(z, tau=0.01), best_z, method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    
    val2 = -res2.fun
    if val2 > best_val:
        best_val = val2
        best_z = res2.x
    
    centers = centers_from_z(best_z)
    
    # 5. Direct coordinate polish
    def objective_direct(centers_flat, tau=0.005):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # Coarse polish
    res3 = minimize(lambda c: objective_direct(c, tau=0.01), centers.ravel(), 
                    method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    centers = res3.x.reshape(N, 3)
    
    # Fine polish
    res4 = minimize(lambda c: objective_direct(c, tau=0.002), centers.ravel(),
                    method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    centers = res4.x.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers
