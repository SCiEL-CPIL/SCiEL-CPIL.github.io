    def _verify(U, V, W, n, m, p):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        
        # Compute full tensor
        # U: (nm, R), V: (mp, R), W: (np, R)
        # T: (nm, mp, np)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check on-support
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        return False
        
        # Check off-support
        # Create a target tensor
        Target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    Target[a, b, c] = 1.0
        
        if not np.allclose(T, Target, atol=1e-9):
            return False
            
        return True
    