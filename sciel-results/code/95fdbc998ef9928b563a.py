<<<<<<< SEARCH
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
=======
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        result = np.zeros((n, m, p))
        for c in range(rank):
            uc = U_np[:, c].reshape(n, m)
            vc = V_np[:, c].reshape(m, p)
            wc = W_np[:, c].reshape(p, n)
            result += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        if np.any(np.abs(result - 1.0) > 1e-9):
            return False
        return True
>>>>>>> REPLACE
