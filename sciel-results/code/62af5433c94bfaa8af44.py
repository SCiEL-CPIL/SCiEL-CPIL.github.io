    def verify(U, V, W, n, m, p, tol=1e-12):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(s - 1) > tol:
                        return False
        return True
