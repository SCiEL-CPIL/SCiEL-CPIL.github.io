def _verify_decomposition_fast(U, V, W, n, m, p, num_tests=3):
    """Verify that the decomposition correctly computes matrix multiplication using vectorized NumPy."""
    import numpy as np
    
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_np = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_np = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    # Use deterministic test cases to avoid any flakiness
    test_cases = [
        np.ones((n, m)),
        np.eye(n, m) if n == m else np.identity(n)[:, :m], # Note: n=2, m=2 so identity
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[1, 1], [1, 1]]),
        np.array([[2, 3], [4, 5]])
    ]
    
    for A in test_cases:
        B = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                      [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]])
        C_expected = A @ B
        
        # Compute C using the decomposition
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_r = U_reshaped[:, :, r]
            V_r = V_reshaped[:, :, r]
            W_r = W_reshaped[:, :, r]
            
            A_U = A * U_r
            B_V = B * V_r
            term = A_U @ B_V
            C_computed += term * W_r.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
