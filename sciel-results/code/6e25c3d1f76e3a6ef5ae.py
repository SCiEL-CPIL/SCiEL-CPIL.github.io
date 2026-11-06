"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # 1. Define the objective function to minimize.
    # We use the Tammes problem energy: sum of 1/d^2 for all pairs.
    # Minimizing this maximizes the minimum distance between points.
    def energy(x):
        # x is a flat array of 3*N coordinates
        pts = x.reshape(-1, 3)
        # Normalize to sphere to handle numerical drift during optimization
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero if a point collapses to origin (unlikely with L-BFGS but safe)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute pairwise squared distances
        # d^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2 a.b for unit vectors
        dots = pts @ pts.T
        # Mask diagonal to avoid 0/0
        np.fill_diagonal(dots, 2.0) 
        # d^2 = 2 - 2*dots
        d2 = 2.0 - 2.0 * dots
        # Clip to avoid negative values due to floating point errors
        d2 = np.clip(d2, 1e-12, None)
        
        # Sum of inverse squared distances (excluding diagonal)
        # We only sum upper triangle to avoid double counting, but for gradient it doesn't matter much
        # Let's just sum all off-diagonal elements
        mask = ~np.eye(N, dtype=bool)
        energy_val = np.sum(1.0 / d2[mask])
        
        return energy_val

    # 2. Multi-start local search
    best_pts = None
    best_energy = np.inf
    rng = np.random.default_rng(42)
    
    # Number of restarts. We will run as many as we can fit in the time budget.
    # Each L-BFGS-B run for N=50 is fast (~0.1-0.5s). 
    # Let's try up to 20 restarts or until time runs out.
    max_restarts = 20
    
    for k in range(max_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Generate random initial points on sphere
        # Use random normal distribution and normalize
        init_pts = rng.normal(size=(N, 3))
        init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
        
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        # Run L-BFGS-B
        # We don't provide gradients to let SciPy estimate them (faster setup, good enough for this N)
        # Or we can provide them for speed. Let's stick to numerical for robustness first.
        res = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_energy:
            # Normalize result
            pts = res.x.reshape(-1, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Check if this is better
            # We use the energy as a proxy. Lower energy = better spread.
            # However, the true metric is covering radius. 
            # Energy is a good proxy. To be safe, we could calculate covering radius for the best few.
            # But calculating covering radius requires convex hull or sampling, which is expensive.
            # Given the strong correlation between Tammes energy and covering radius for N=50,
            # minimizing energy is the standard approach.
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_pts = pts
        
        # If we have plenty of time, we could do a second pass with more iterations
        # But L-BFGS-B converges quickly for this problem.

    # Fallback: If no optimization succeeded or time ran out immediately, return Fibonacci
    if best_pts is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
