"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Initial Hexagonal Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    x0 = None
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Ensure points are well inside to allow movement
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    def objective(x, beta):
        x = x.reshape(-1, 2)
        # Distance to boundary
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        # Pairwise distances divided by 2
        d_pair = pdist(x) / 2.0
        # Combine all "radius constraints"
        d = np.concatenate([dist_boundary, d_pair])
        
        # Soft-min approximation: 
        # We want to maximize min(d).
        # Minimize -min(d).
        # Surrogate: (1/beta) * log(sum(exp(-beta * d)))
        # This function is smooth and approximates -min(d) as beta -> inf.
        
        A = -beta * d
        max_A = np.max(A)
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta

    best_x = x0.copy()
    best_val = objective(best_x, 100) # Arbitrary beta for comparison
    
    # Annealing schedule
    betas = [10, 50, 100, 200, 500]
    
    # Optimization loop
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(objective, best_x, args=(beta,), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Check if result is better
            val = res.fun
            if val < best_val:
                best_x = res.x.copy()
                best_val = val
        except Exception:
            pass

    # If time remains, try a few perturbations from the best solution
    if time.time() < deadline - 20:
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            x_perturbed = best_x + np.random.randn(2 * N) * 0.01
            x_perturbed = np.clip(x_perturbed, -0.99, 0.99)
            
            try:
                # Use a high beta for local polish
                res = minimize(objective, x_perturbed, args=(500,), method='L-BFGS-B',
                               bounds=[(-1.0, 1.0)] * (2 * N),
                               options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < best_val:
                    best_x = res.x.copy()
                    best_val = res.fun
            except Exception:
                pass

    # Final validation and cleanup
    best_x = best_x.reshape(-1, 2)
    # Ensure no points are outside (clip if slightly outside due to numerical error)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        # Scale back to inside
        best_x[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    return best_x
# EVOLVE-BLOCK-END
