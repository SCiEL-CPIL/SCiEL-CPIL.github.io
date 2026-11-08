def _refine_ah(U, V, W, n, m, p, iters=10):
    r = len(U[0])
    
    def round_half(x):
        return round(x * 2) / 2.0

    for _ in range(iters):
        if _remaining() < 5.0:
            break
            
        # Round U
        U_r = [[round_half(v) for v in row] for row in U]
        
        # Update W (given U_r, V)
        for i in range(n):
            for k in range(p):
                A = np.array([[U_r[i * m + j][c] * V[j * p + k][c] for c in range(r)] for j in range(m)])
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass

        # Update V (given U_r, W)
        for j in range(m):
            for k in range(p):
                A = np.array([[U_r[i * m + j][c] * W[k * n + i][c] for c in range(r)] for i in range(n)])
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
        
        # Round V
        V_r = [[round_half(v) for v in row] for row in V]
        
        # Update U (given V_r, W)
        for i in range(n):
            for j in range(m):
                A = np.array([[V_r[j * p + k][c] * W[k * n + i][c] for c in range(r)] for k in range(p)])
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass

        # Update W (given U, V_r)
        for i in range(n):
            for k in range(p):
                A = np.array([[U[i * m + j][c] * V_r[j * p + k][c] for c in range(r)] for j in range(m)])
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass

        # Round W
        W_r = [[round_half(v) for v in row] for row in W]
        
        # Update U (given V_r, W_r)
        for i in range(n):
            for j in range(m):
                A = np.array([[V_r[j * p + k][c] * W_r[k * n + i][c] for c in range(r)] for k in range(p)])
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass

        # Update V (given U, W_r)
        for j in range(m):
            for k in range(p):
                A = np.array([[U[i * m + j][c] * W_r[k * n + i][c] for c in range(r)] for i in range(n)])
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass

    return U, V, W
