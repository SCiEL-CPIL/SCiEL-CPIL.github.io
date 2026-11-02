def _find_rank_10_2x2x3():
    # T is 4x6x6 with 12 non-zero entries?
    # No, 2x2x3 has n=2, m=2, p=3.
    # Non-zero entries: i in 0..1, j in 0..1, k in 0..2.
    # Total 2*2*3 = 12 entries.
    # U: 4x10, V: 6x10, W: 6x10.
    # 100 variables.
    # 144 equations (4*6*6).
    # We want T_recon = T.
    
    T = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T[i*2+j, j*3+k, k*2+i] = 1.0

    # Use random restarts
    for seed in range(10):
        np.random.seed(seed)
        U = np.random.randn(4, 10)
        V = np.random.randn(6, 10)
        W = np.random.randn(6, 10)
        
        # Scale down
        U *= 0.1
        V *= 0.1
        W *= 0.1
        
        # Use scipy.optimize.minimize on the flattened parameters
        # This might be slow.
        
        # Alternative: Gradient descent with momentum
        lr = 0.01
        for step in range(2000):
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = T - T_recon
            if np.linalg.norm(err) < 1e-10:
                return U, V, W
            
            dU = 2 * np.einsum('ijk,jr,kr->ir', err, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', err, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Normalize
            U /= np.linalg.norm(U) + 1e-9
            V /= np.linalg.norm(V) + 1e-9
            W /= np.linalg.norm(W) + 1e-9
            
    return None
