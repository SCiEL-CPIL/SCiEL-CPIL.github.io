import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.special import eval_hermite
import math

def construct():
    indices = [0, 4, 8, 12]
    n_vars = len(indices)
    
    # 1. Compute N
    N = np.zeros((n_vars, n_vars))
    for i, k in enumerate(indices):
        N[i, i] = np.sqrt(np.pi) * (2 ** k) * math.factorial(k)
        
    # 2. Compute M
    M = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            def integrand(x):
                hi = eval_hermite(indices[i], x)
                hj = eval_hermite(indices[j], x)
                return (x ** 4) * hi * hj * np.exp(-x ** 2)
            
            val, err = quad(integrand, -np.inf, np.inf, limit=200)
            M[i, j] = val
            
    # 3. Constraint h
    h = np.array([eval_hermite(k, 0) for k in indices])
    
    # 4. Basis B
    B = linalg.null_space(h.reshape(1, -1))
    
    # 5. Project
    M_proj = B.T @ M @ B
    N_proj = B.T @ N @ B
    
    # 6. Eigenvalues
    eigenvalues, eigenvectors = linalg.eig(M_proj, N_proj)
    
    # 7. Min real eigenvalue
    real_mask = np.abs(eigenvalues.imag) < 1e-9
    real_eigenvalues = eigenvalues[real_mask].real
    real_eigenvectors = eigenvectors[:, real_mask]
    
    if len(real_eigenvalues) == 0:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
        
    min_idx = np.argmin(real_eigenvalues)
    y = real_eigenvectors[:, min_idx]
    c = B @ y
    
    return np.array([c[0], c[1], c[2]])
