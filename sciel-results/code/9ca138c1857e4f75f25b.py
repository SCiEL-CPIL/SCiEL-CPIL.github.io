<<<<<<< SEARCH
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
=======
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is valid and finite."""
        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
            return False
        r = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if not np.isfinite(s):
                        return False
                    if abs(s - 1.0) > tol:
                        return False
        return True
>>>>>>> REPLACE

