import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # r ~ 1/sqrt(N).
    # Spacing ~ 2r ~ 2/sqrt(N).
    spacing = 2.0 / np.sqrt(N) * 0.9
    points = []
    # Generate lattice points in a large box
    k_max = int(3.0 / spacing) + 5
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0 - spacing / 2:
                points.append([x, y])
    
    points = np.array(points)
    # If we don't have enough, reduce spacing
    while len(points) < N:
        spacing *= 0.95
        points = []
        k_max = int(3.0 / spacing) + 5
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = i * spacing + 0.5 * spacing * (j % 2)
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - spacing / 2:
                    points.append([x, y])
        points = np.array(points)
        
    # Select N points closest to center (usually best for packing)
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)
    x0 = points[indices[:N]]
    
    # 2. Optimization
    k = 20.0  # Sharpness parameter for soft-min
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        
        # Weights for pairwise term
        # W_ij = exp(-k * dist_ij)
        # We only need i < j for the sum S, but for gradient we need all j != i
        # Let's compute the full matrix
        W = np.exp(-k * dist)
        np.fill_diagonal(W, 0.0)
        
        # Sum S
        S = np.sum(W)
        
        # Gradient of pairwise part
        # G_pair_ik = -1/S * sum_j W_ij * (x_ik - x_jk) / dist_ij
        # V_ij = W_ij / dist_ij
        V = W / dist_safe
        np.fill_diagonal(V, 0.0)
        
        # sum_j V_ij * x_jk is V @ x
        # sum_j V_ij is V.sum(axis=1)
        sum_V = V.sum(axis=1)
        Vx = V @ x
        
        grad_pair = (x * sum_V[:, np.newaxis] - Vx) / S
        grad_pair *= -1.0
        
        # Boundary term
        r = np.hypot(x[:, 0], x[:, 1])
        r_safe = np.where(r > 1e-10, r, 1.0)
        
        # T = sum exp(-k * (1 - r_i))
        boundary_dist = 1.0 - r
        T_weights = np.exp(-k * boundary_dist)
        T = np.sum(T_weights)
        
        # Gradient of boundary part
        # G_bound_ik = 1/T * T_weights_i * x_ik / r_i
        grad_bound = (x / r_safe[:, np.newaxis]) * (T_weights / T)[:, np.newaxis]
        
        # Total objective
        obj = (1.0 / k) * (np.log(S) + np.log(T))
        
        # Total gradient
        grad = grad_pair + grad_bound
        
        return obj, grad.reshape(-1)

    # Run optimization
    result = None
    try:
        # Use L-BFGS-B
        res = minimize(objective_and_grad, x0.flatten(), method='L-BFGS-B', 
                       jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() - start_time < time_limit:
            result = res.x.reshape(N, 2)
    except Exception:
        result = x0

    # Final check and return
    if result is None:
        result = x0
        
    # Ensure points are inside
    radii = np.hypot(result[:, 0], result[:, 1])
    mask = radii <= 1.0
    if not np.all(mask):
        # Clip or fallback
        result = x0

    return result
