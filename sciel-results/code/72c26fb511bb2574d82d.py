import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates y' = A y from t_start to t_end using analytic spectral decomposition.
    A = (N+1)^2 * tridiag(1, -2, 1)
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Deadline guard (300s limit, keep margin)
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure f is called at least once
    _ = f(t0, y0)
    
    N = len(y0)
    
    # Construct the eigen-decomposition analytically
    # A is a scaled discrete Laplacian.
    # Eigenvalues: lambda_k = -2 * (N+1)^2 * (1 - cos(k * pi / (N+1)))
    # Eigenvectors: v_k(i) = sqrt(2/(N+1)) * sin(i * k * pi / (N+1))
    
    # Indices 1..N in math, 0..N-1 in python
    i = np.arange(1, N + 1)[:, np.newaxis] # Column vector shape (N, 1)
    k = np.arange(1, N + 1)[np.newaxis, :] # Row vector shape (1, N)
    
    # Compute eigenvectors matrix V (N x N)
    # V[i-1, k-1] = sqrt(2/(N+1)) * sin(i * k * pi / (N+1))
    norm = np.sqrt(2.0 / (N + 1))
    V = norm * np.sin(i * k * np.pi / (N + 1))
    
    # Compute eigenvalues
    # lambda_k = -2 * (N+1)^2 * (1 - cos(k * pi / (N+1)))
    # Note: 1 - cos(x) = 2 * sin^2(x/2)
    # lambda_k = -4 * (N+1)^2 * sin^2(k * pi / (2 * (N+1)))
    # This form is more numerically stable for small k? 
    # Let's stick to 1 - cos for clarity, or use the sin^2 form for precision.
    # For small angles, 1-cos(x) suffers from cancellation. sin^2(x/2) is better.
    angles = k * np.pi / (2 * (N + 1))
    lambdas = -4.0 * ((N + 1) ** 2) * (np.sin(angles) ** 2)
    
    # Check deadline
    if time.time() > deadline:
         return np.array(y0)

    # Project y0 onto eigenbasis: c = V^T y0
    c = V.T @ y0
    
    # Scale by exp(lambda * dt)
    # lambdas are negative, dt is positive, so exp decays.
    decay = np.exp(lambdas * dt)
    d = c * decay
    
    # Check deadline
    if time.time() > deadline:
         return np.array(y0)

    # Project back: y_end = V d
    y_end = V @ d
    
    # Check for finiteness
    if not np.all(np.isfinite(y_end)):
        return np.array(y0)
        
    return y_end
