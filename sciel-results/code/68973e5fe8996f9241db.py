import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Create a dense set of test points on the sphere for approximating covering radius
    # Use a Fibonacci spiral for test points too, with more points
    N_test = 20000
    i_t = np.arange(N_test) + 0.5
    z_t = 1.0 - 2.0 * i_t / N_test
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * i_t
    r_t = np.sqrt(1.0 - z_t * z_t)
    Y = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    # Precompute Y as rows for broadcasting
    # Y shape: (N_test, 3)
    # P shape: (N, 3)
    
    # Optimization parameters
    eta = 0.01  # Learning rate
    n_iter = 5000  # Max iterations
    top_k = 50  # Number of "worst" test points to consider for gradient
    
    best_covering_radius = np.inf
    best_P = P.copy()
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        # Compute distances from all test points Y to all points P
        # D[i, j] is distance between Y[i] and P[j]
        # Use dot product for efficiency: d = sqrt(2 - 2 * Y @ P.T)
        # Y @ P.T shape: (N_test, N)
        dots = Y @ P.T
        # Clip to avoid numerical issues
        dots = np.clip(dots, -1.0, 1.0)
        D = np.sqrt(2.0 - 2.0 * dots)
        
        # Find nearest point for each test point
        min_D = np.min(D, axis=1)
        nearest_idx = np.argmin(D, axis=1)
        
        # Current covering radius approximation
        current_radius = np.max(min_D)
        
        if current_radius < best_covering_radius:
            best_covering_radius = current_radius
            best_P = P.copy()
            
        # Select top_k worst test points
        # We want the largest min_D values
        worst_indices = np.argpartition(min_D, -top_k)[-top_k:]
        worst_D = min_D[worst_indices]
        worst_nearest = nearest_idx[worst_indices]
        worst_Y = Y[worst_indices]
        
        # Compute subgradient for each point P[k]
        # grad[k] is a vector in R^3
        grad = np.zeros_like(P)
        
        for idx in range(top_k):
            y = worst_Y[idx]
            k = worst_nearest[idx]
            p_k = P[k]
            d = worst_D[idx]
            if d < 1e-6:
                continue
            # Gradient of distance d wrt p_k is -(y - p_k) / d
            # We want to decrease the distance, so we move p_k towards y
            # Direction is (y - p_k) / d
            dir_vec = (y - p_k) / d
            grad[k] += dir_vec
            
        # Project gradient onto tangent space of sphere at P[k]
        # Tangent space is orthogonal to P[k]
        # grad_tangent[k] = grad[k] - (grad[k] . P[k]) * P[k]
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        
        # Add a small repulsive force to prevent clustering
        # Compute pairwise distances
        # This is O(N^2) which is fine for N=130
        P_norm = P / np.linalg.norm(P, axis=1, keepdims=True)
        dots_P = P_norm @ P_norm.T
        np.fill_diagonal(dots_P, 1.0)
        D_P = np.sqrt(np.clip(2.0 - 2.0 * dots_P, 0.0, 4.0))
        
        # Repulsive force: 1 / D_P^2, zero diagonal
        rep_force = np.zeros_like(P)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d_ij = D_P[i, j]
                    if d_ij < 0.01:
                        f = 1.0 / (d_ij * d_ij)
                    else:
                        f = 1.0 / (d_ij * d_ij)
                    # Force is in direction P[i] - P[j]
                    rep_force[i] += f * (P[i] - P[j])
        
        # Combine gradient and repulsive force
        # Scale repulsive force down
        rep_scale = 0.001
        total_grad = grad_tangent + rep_scale * rep_force
        
        # Project total_grad onto tangent space again (just in case)
        total_grad_dot_P = np.sum(total_grad * P, axis=1, keepdims=True)
        total_grad_tangent = total_grad - total_grad_dot_P * P
        
        # Update P
        P = P + eta * total_grad_tangent
        
        # Normalize P to unit sphere
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        
    return best_P
