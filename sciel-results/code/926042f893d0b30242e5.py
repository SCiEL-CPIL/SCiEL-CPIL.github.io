import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0

    dt = t1 - t0
    if dt == 0:
        return y0.copy()
    
    # If y0 is zero, return zero
    norm_y0 = np.linalg.norm(y0)
    if norm_y0 < 1e-15:
        return np.zeros_like(y0)

    # Krylov Subspace Method (Arnoldi Iteration)
    # We build the Krylov subspace K_m(A, y0)
    # v_0 = y0 / ||y0||
    # v_{j+1} = (A v_j - sum h_{ij} v_i) / h_{j+1,j}
    
    max_k = n  # Maximum possible dimension
    k = 0
    V = np.zeros((n, max_k), dtype=float)
    H = np.zeros((max_k, max_k), dtype=float)
    
    # Initialize
    v = y0 / norm_y0
    V[:, 0] = v
    k = 1
    
    for j in range(max_k - 1):
        if time.time() > deadline:
            # Fallback to partial result or simple return if time runs out
            # But 15 calls is very fast, so this is unlikely to trigger
            break
            
        # Apply A to v_j: w = A @ V[:, j]
        # Since f(t, y) = A y (linear, time independent assumed based on task "stiff linear system")
        # We evaluate at t0.
        w = f(t0, V[:, j])
        
        # Modified Gram-Schmidt Orthogonalization
        # Project w onto V[:, 0..j]
        for i in range(j + 1):
            h_ij = np.dot(V[:, i], w)
            H[i, j] = h_ij
            w -= h_ij * V[:, i]
            
        # Second pass for numerical stability (reorthogonalization)
        for i in range(j + 1):
            h_ij = np.dot(V[:, i], w)
            H[i, j] += h_ij
            w -= h_ij * V[:, i]
            
        h_jj = np.linalg.norm(w)
        H[j + 1, j] = h_jj
        
        if h_jj < 1e-14:
            # Breakdown: Krylov subspace is exhausted
            break
            
        V[:, j + 1] = w / h_jj
        k += 1
        
    # We have V (n x k) and H (k x k, upper Hessenberg)
    # The solution is y(t) = V @ expm(dt * H) @ (V.T @ y0)
    
    # Compute coordinates of y0 in the Krylov basis
    # y0 = V @ c, where c = V.T @ y0 (since V is orthonormal)
    c = V[:, :k].T @ y0
    
    # Solve the small ODE in the Krylov subspace
    # u' = H u, u(0) = c
    # u(dt) = expm(dt * H) @ c
    H_k = H[:k, :k]
    u_final = expm(dt * H_k) @ c
    
    # Map back to original space
    y_final = V[:, :k] @ u_final
    
    return y_final
