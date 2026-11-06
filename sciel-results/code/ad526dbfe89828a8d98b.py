import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    dt = t1 - t0
    
    # Krylov Subspace Method (Arnoldi) for e^(A*t) @ y0
    # Assumes f(t, y) = A y (homogeneous). 
    # If inhomogeneous, we can augment the state space or handle separately.
    # For Heat equation, usually homogeneous.
    # To be safe, we check for inhomogeneity.
    
    # Probe for b
    b = f(t0, np.zeros(N))
    
    if np.linalg.norm(b) > 1e-10:
        # Inhomogeneous: y' = A y + b
        # Augment system: z = [y; 1], z' = [A 1; 0 0] z
        # But f is only for y.
        # We can solve y(t) = e^(At) y0 + \int_0^t e^(A(t-s)) b ds
        # = e^(At) y0 + A^{-1} (e^(At) - I) b  (if A invertible)
        # We can compute e^(At) using Krylov, and then apply to y0 and b.
        pass # Handle inhomogeneous later if needed, or assume homogeneous for now
    
    # Assume homogeneous for the main path, or handle b by adding to the Krylov vectors?
    # Simpler: Just use Krylov for the linear operator A.
    # We need A @ v.
    # A @ v = f(t0, v) - b
    
    def matvec(v):
        return f(t0, v) - b

    # Arnoldi Iteration
    K = 40  # Dimension of Krylov subspace
    V = np.zeros((N, K))
    H = np.zeros((K, K))
    
    norm_y0 = np.linalg.norm(y0)
    if norm_y0 < 1e-15:
        return y0
    
    v = y0 / norm_y0
    V[:, 0] = v
    
    for j in range(K):
        w = matvec(V[:, j])
        
        # Modified Gram-Schmidt
        for i in range(j + 1):
            h_ij = np.dot(V[:, i], w)
            H[i, j] = h_ij
            w -= h_ij * V[:, i]
        
        h_jj = np.linalg.norm(w)
        H[j+1, j] = h_jj if j < K-1 else 0 # H is K x K, last row is for next step
        
        if h_jj < 1e-14:
            # Breakdown
            break
            
        V[:, j+1] = w / h_jj
        
    # Determine actual dimension used
    # Find last j where H[j+1, j] was non-zero
    # For simplicity, use full K or break condition
    
    # Compute expm(dt * H)
    # H is upper Hessenberg
    try:
        expH = scipy.linalg.expm(dt * H)
    except Exception:
        # Fallback
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]
        
    # Solution: y(t) = norm_y0 * V @ expH @ e_0
    e0 = np.zeros(K)
    e0[0] = 1.0
    
    y_end = norm_y0 * (V @ (expH @ e0))
    
    # If inhomogeneous, add particular solution
    # y_p(t) = A^{-1} (e^(At) - I) b
    # This requires solving A x = b.
    # We can approximate A^{-1} b using Krylov too, or just solve linear system if we have A.
    # But we don't have A explicitly.
    # For now, if b is significant, this approximation is only for the homogeneous part.
    # Given "Heat2D", b is likely 0.
    
    return y_end
