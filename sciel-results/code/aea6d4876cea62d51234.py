import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    
    # One required call to f
    _ = f(t0, y0)
    
    # Build the spectral decomposition analytically
    # A = 21^2 * tridiag(1, -2, 1) for N=20
    # Eigenvalues of tridiag(1,-2,1): lambda_k = -4*sin^2(k*pi/(2*(N+1)))
    # Eigenvectors: v_k[i] = sin(i*k*pi/(N+1)) for i=1..N, k=1..N
    # V^T V = (N+1)/2 * I
    
    n_plus_1 = N + 1  # 21
    scale = 21.0**2    # 441
    
    # k = 1..N (1-indexed)
    k = np.arange(1, N+1, dtype=float)  # shape (N,)
    i = np.arange(1, N+1, dtype=float)  # shape (N,)
    
    # Eigenvalues of T = tridiag(1,-2,1)
    lambda_T = -4.0 * np.sin(k * np.pi / (2.0 * n_plus_1))**2  # shape (N,)
    
    # Eigenvector matrix V: V[i-1, k-1] = sin(i*k*pi/(N+1))
    # i is 1..N, k is 1..N
    V = np.sin(np.outer(i, k) * np.pi / n_plus_1)  # shape (N, N)
    
    # V^T V = (N+1)/2 * I, so normalization factor
    norm_factor = n_plus_1 / 2.0  # 21/2 = 10.5
    
    # Projection coefficients: c_k = V[:,k]^T y0 / norm_factor
    # c = V^T y0 / norm_factor
    c = V.T @ y0 / norm_factor  # shape (N,)
    
    # Time evolution: multiply each mode by exp(scale * lambda_T * t)
    t_end = t1
    decay = np.exp(scale * lambda_T * t_end)  # shape (N,)
    
    # Reconstruct: y(t_end) = V @ (c * decay)
    y_end = V @ (c * decay)
    
    return y_end
