    def verify_decomp(U, V, W, rank):
        # T[k, i, j] should be 1 if C[k,i] = sum A[i,j]B[j,k]
        # Check all k, i, j
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
