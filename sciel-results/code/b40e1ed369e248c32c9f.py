<<<<<<< SEARCH
            # Update V: for each (j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = U[:n*m, :].reshape(n, m, r)[:, j, :] * W[np.arange(p) * n + k][:, np.newaxis]
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    M = U[np.arange(n) * m + j, :] * W[np.arange(p) * n + k][:, np.newaxis].T
                    # Wait, I need M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # U[i*m+j, r] for i=0..n-1, r=0..r-1: shape (n, r)
                    # W[k*n+i, r] for i=0..n-1, r=0..r-1: shape (n, r)
                    # M[i, r] = product: shape (n, r)
                    M = U[np.arange(n) * m + j, :] * W[np.arange(n) * k + k * 0 + np.arange(n) * n + k][:, np.newaxis]
                    # Hmm, W[k*n+i, r] where i ranges 0..n-1
                    # So W indices are: k*n+0, k*n+1, ..., k*n+(n-1)
                    W_idx = np.arange(n) * 1 + k * n  # k*n + i for i in 0..n-1
                    M = U[np.arange(n) * m + j, :] * W[W_idx, :].T
                    # Now M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # We need: sum_r M[i,r] * V[j*p+k,r] = 1 for all i
                    # i.e., M @ V[j*p+k, :]^T = ones(n)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_r[j*p+k, :] = x
                    except:
                        pass
=======
            # Update V: for each (j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    U_idx = np.arange(n) * m + j  # i*m + j for i in 0..n-1
                    W_idx = k * n + np.arange(n)  # k*n + i for i in 0..n-1
                    M = U[U_idx, :] * W[W_idx, :].T
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # We need: sum_r M[i,r] * V[j*p+k,r] = 1 for all i
                    # i.e., M @ V[j*p+k, :]^T = ones(n)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_r[j*p+k, :] = x
                    except:
                        pass
>>>>>>> REPLACE

