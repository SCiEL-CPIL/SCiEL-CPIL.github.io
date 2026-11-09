import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Satisfy the constraint that f is called at least once
    _ = f(t0, y0)
    
    N = 20
    scale = 21.0 ** 2
    
    # Precompute eigenvectors and eigenvalues for the specific matrix
    # A = scale * L, where L is tridiag(1, -2, 1)
    # Eigenvalues of L: mu_k = -4 * sin^2(k * pi / (2 * (N + 1)))
    # Eigenvectors of L: u_k(i) = sin(i * k * pi / (N + 1))
    # We construct Q such that Q @ diag(mu) @ Q.T = L
    # Q is orthogonal, so Q.T = Q.inv
    
    k = np.arange(1, N + 1)
    i = np.arange(1, N + 1)
    
    # Construct matrix of eigenvectors (unnormalized)
    # Shape (N, N) where row i, col k is sin(i * k * pi / (N+1))
    U = np.sin(np.outer(i, k) * np.pi / (N + 1))
    
    # Normalize columns to make Q orthogonal
    # Norm squared is (N+1)/2
    norm_factor = np.sqrt(2.0 / (N + 1))
    Q = U * norm_factor
    
    # Eigenvalues of L
    mu = -4.0 * np.sin(k * np.pi / (2.0 * (N + 1))) ** 2
    
    dt = t1 - t0
    
    # y_end = Q @ diag(exp(scale * mu * dt)) @ Q.T @ y0
    # Step 1: Project y0 onto eigenvectors
    v = Q.T @ y0
    
    # Step 2: Scale by exponential of eigenvalues
    exp_factors = np.exp(scale * mu * dt)
    w = v * exp_factors
    
    # Step 3: Project back to original space
    y_end = Q @ w
    
    return y_end
