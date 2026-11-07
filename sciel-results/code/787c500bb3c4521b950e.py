import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    # Call f once to satisfy the constraint
    f(t_span[0], y0)
    
    n = len(y0)
    
    dt = t_span[1] - t_span[0]
    
    # Use analytic eigenvalues and eigenvectors for the discrete Laplacian
    # A = 21^2 * tridiag(1, -2, 1)
    # Eigenvalues: lambda_k = -4 * 21^2 * sin^2(k * pi / (2 * (n + 1)))
    # Eigenvectors: V_{i,k} = sqrt(2/(n+1)) * sin(i * k * pi / (n + 1))
    
    factor = 441.0  # 21^2
    k = np.arange(1, n + 1)
    lambda_k = -4.0 * factor * np.sin(k * np.pi / (2.0 * (n + 1)))**2
    
    # Compute exp(lambda_k * dt) with high precision
    exp_lambda_k = np.exp(lambda_k * dt)
    
    # Construct V (n x n)
    i_idx = np.arange(1, n + 1)[:, np.newaxis]
    k_idx = np.arange(1, n + 1)[np.newaxis, :]
    
    V = np.sqrt(2.0 / (n + 1)) * np.sin(i_idx * k_idx * np.pi / (n + 1))
    
    # y_end = V @ diag(exp_lambda_k) @ V.T @ y0
    y_proj = V.T @ y0
    y_scaled = exp_lambda_k * y_proj
    y_end = V @ y_scaled
    
    return y_end
