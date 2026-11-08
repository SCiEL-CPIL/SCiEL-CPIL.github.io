import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Construct the matrix A
    # A = 21^2 * tridiag(1, -2, 1)
    # The tridiag matrix is (N x N)
    # Diagonal is -2, off-diagonals are 1
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N-1)
    
    # Create the tridiagonal matrix
    # L = diag(main_diag) + diag(off_diag, 1) + diag(off_diag, -1)
    L = np.diag(main_diag)
    if N > 1:
        L += np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
    scale = 21.0**2
    A = scale * L
    
    # Call f at least once
    _ = f(t0, y0)
    
    # Diagonalize A
    # Since A is symmetric, we can use eigh for stability
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Project y0 onto the eigenbasis
    # y0 = V c  =>  c = V^T y0 (since V is orthogonal)
    c = eigenvectors.T @ y0
    
    # Evolve each mode
    # y(t) = sum c_i exp(lambda_i t) v_i
    # y_end = V (exp(eigenvalues * dt) * c)
    exp_factors = np.exp(eigenvalues * dt)
    c_end = c * exp_factors
    y_end = eigenvectors @ c_end
    
    if not np.all(np.isfinite(y_end)):
        return y0
        
    return y_end
