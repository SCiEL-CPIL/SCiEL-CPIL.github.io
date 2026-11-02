def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 12, 32, 24
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to find lower-rank decompositions
    for r in range(32, 97):
        if time.time() > deadline - 15:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for iteration in range(100):
                if time.time() > deadline - 2:
                    break
                
                # Solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        row_idx = i * m + j
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        b = np.ones(p)
                        U_new[row_idx, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                U = U_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        row_idx = j * p + k
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        b = np.ones(n)
                        V_new[row_idx, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                V = V_new
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        row_idx = k * n + i
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        b = np.ones(m)
                        W_new[row_idx, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                W = W_new
            
            if verify(U, V, W, r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    r = n * m * p
    U = [[0] * r for _ in range(nm)]
    V = [[0] * r for _ in range(mp)]
    W = [[0] * r for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
