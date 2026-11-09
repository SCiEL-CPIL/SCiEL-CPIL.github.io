import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe within 120s limit

    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be safe
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    def energy_and_grad(P):
        # Compute pairwise distances
        # P is (N, 3)
        # dist^2[i,j] = 2 - 2 * P[i] dot P[j]
        dots = np.dot(P, P.T)
        d2 = 2.0 - 2.0 * dots
        # Avoid division by zero on diagonal
        np.fill_diagonal(d2, 1.0)
        
        # Energy: Sum of 1/d2 for i < j
        # Using 1/d2 is a good repulsive potential
        # We can also use 1/d2^2 for stronger repulsion
        inv_d2 = 1.0 / d2
        inv_d2 -= np.diag(np.diag(inv_d2)) # Zero out diagonal
        
        E = np.sum(inv_d2) / 2.0
        
        # Gradient of E w.r.t P[i]
        # d/dP_i (1/d2_ij) = d/dP_i (1 / (2 - 2 P_i.P_j))
        # = -1 / (2 - 2 P_i.P_j)^2 * d/dP_i (2 - 2 P_i.P_j)
        # = -1 / d2_ij^2 * (-2 P_j) = 2 P_j / d2_ij^2
        # Sum over j != i
        
        # inv_d2 is symmetric. Let W = inv_d2.
        # Grad_i = sum_{j != i} 2 * P_j * W[i,j]^2 ? 
        # Let's re-derive carefully.
        # E = 0.5 * sum_{i!=j} 1/d2_ij
        # dE/dP_i = sum_{j!=i} d(1/d2_ij)/dP_i
        # d(1/d2)/dP_i = -1/d2^2 * d(d2)/dP_i
        # d(d2)/dP_i = -2 P_j
        # So term is 2 P_j / d2_ij^2
        # W[i,j] = 1/d2_ij
        # Term is 2 * P_j * W[i,j]^2
        
        W2 = inv_d2 * inv_d2
        grad = np.zeros_like(P)
        for i in range(N):
            # sum_j 2 * P_j * W2[i,j]
            # We can do this with matrix multiplication
            # grad[i] = 2 * (W2[i, :] @ P)
            pass
            
        # Vectorized:
        # grad_i = 2 * sum_j W2[i,j] P_j
        # grad = 2 * (W2 @ P)
        grad = 2.0 * (W2 @ P)
        
        # Project gradient to tangent space
        # P_i is unit vector. Tangent space is orthogonal to P_i.
        # grad_tan = grad - (P_i dot grad) P_i
        proj = np.sum(P * grad, axis=1, keepdims=True)
        grad_tan = grad - proj * P
        
        return E, grad_tan

    # Optimization Loop
    # We use a simple gradient descent with cooling
    # Initial step size
    eta = 0.01
    min_eta = 1e-6
    cooling_rate = 0.999
    
    best_P = P.copy()
    best_E, _ = energy_and_grad(P)
    
    while time.time() < deadline:
        E, grad_tan = energy_and_grad(P)
        
        # Update
        step = eta * grad_tan
        P_new = P + step
        
        # Normalize back to sphere (simple projection, good enough for small steps)
        P_new /= np.linalg.norm(P_new, axis=1, keepdims=True)
        
        # Check if energy decreased
        E_new, _ = energy_and_grad(P_new)
        
        if E_new < best_E:
            best_E = E_new
            best_P = P_new.copy()
            
        if E_new < E:
            P = P_new
        else:
            # If it increased, reduce step size or revert
            # Simple approach: revert and cool down
            P = P.copy() # revert to previous
            eta *= 0.9
            
        # Cool down
        eta = max(eta * cooling_rate, min_eta)
        
        # Early stopping if convergence
        if eta < min_eta * 1.01:
            break

    return best_P
# EVOLVE-BLOCK-END
