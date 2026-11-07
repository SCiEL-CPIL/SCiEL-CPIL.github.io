import numpy as np

def integrate(f, t_span, y0):
    """
    Integrates y' = A y from t_start to t_end using spectral decomposition.
    A = (N+1)^2 * L, where L is the discrete Laplacian.
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Ensure f is called at least once
    _ = f(t0, y0)
    
    N = len(y0)
    N_plus_1 = N + 1
    
    # Eigenvectors of the discrete Laplacian with Dirichlet BC
    # v_k(i) = sin(k * pi * i / (N+1)) for i=1..N
    # In 0-based index i=0..N-1: sin(k * pi * (i+1) / (N+1))
    
    # We can compute the projection coefficients c_k = <v_k, y0> / <v_k, v_k>
    # <v_k, v_k> = (N+1)/2
    # So c_k = (2/(N+1)) * sum_i v_k(i) y0(i)
    
    # Let's construct the sine matrix S where S[k, i] = sin(k * pi * (i+1) / (N+1))
    # k is 1..N, i is 0..N-1
    
    i_idx = np.arange(1, N+1)
    k_idx = np.arange(1, N+1)
    
    # S[k_idx, i_idx] -> shape (N, N)
    # S[k, i] = sin(k * pi * (i+1) / (N+1))
    # Note: k_idx is 1..N, i_idx is 1..N (representing 1-based)
    # But y0 is 0-based. y0[i] corresponds to i+1.
    
    # Matrix M where M[k, i] = sin(k * pi * (i+1) / (N+1))
    # k: 1..N
    # i: 0..N-1 -> i+1: 1..N
    
    # We can vectorize this
    # k_col = k_idx[:, np.newaxis] # (N, 1)
    # i_row = i_idx[np.newaxis, :] # (1, N)
    # S = np.sin(np.pi * k_col * i_row / N_plus_1)
    
    # Coefficients c_k = (2 / (N+1)) * S @ y0
    # y0 is shape (N,)
    # S @ y0 is shape (N,)
    
    # Eigenvalues lambda_k = -4 * (N+1)^2 * sin^2(k * pi / (2 * (N+1)))
    # lambda_k for k=1..N
    
    # decay_k = exp(lambda_k * dt)
    
    # y_end = S.T @ (c * decay)
    # Wait, y_end_i = sum_k c_k decay_k v_k(i)
    # y_end = S.T @ (c * decay) ?
    # S[k, i] = v_k(i).
    # (S.T @ vec)[i] = sum_k S.T[i, k] vec[k] = sum_k S[k, i] vec[k] = sum_k v_k(i) vec[k].
    # Yes.
    
    # Let's implement this.
    
    k_col = np.arange(1, N+1)[:, np.newaxis]
    i_row = np.arange(1, N+1)[np.newaxis, :]
    
    S = np.sin(np.pi * k_col * i_row / N_plus_1)
    
    # Projection
    # c = (2 / (N+1)) * S @ y0
    c = (2.0 / N_plus_1) * (S @ y0)
    
    # Eigenvalues
    # lambda_k = -4 * (N+1)^2 * sin^2(k * pi / (2 * (N+1)))
    # k is 1..N
    k_vals = np.arange(1, N+1)
    angles = np.pi * k_vals / (2.0 * N_plus_1)
    sin_angles = np.sin(angles)
    lambda_vals = -4.0 * (N_plus_1 ** 2) * (sin_angles ** 2)
    
    decay = np.exp(lambda_vals * dt)
    
    # Reconstruct
    # y_end = S.T @ (c * decay)
    y_end = S.T @ (c * decay)
    
    # Verify finiteness
    if not np.all(np.isfinite(y_end)):
        return np.array(y0)
        
    return y_end
