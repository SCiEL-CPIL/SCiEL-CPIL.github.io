   def _als_decompose_fast(n, m, p, rank, max_iters=100, tol=1e-10):
       U = np.random.randn(n * m, rank)
       V = np.random.randn(m * p, rank)
       W = np.random.randn(p * n, rank)
       
       # Reshape for easier indexing
       # U: (n, m, R)
       # V: (m, p, R)
       # W: (p, n, R)
       
       for it in range(max_iters):
           # Update U
           # For each i, j: solve (V[j] * W[:, i]) u = 1_p
           U_new = np.zeros_like(U)
           for i in range(n):
               for j in range(m):
                   # V[j] is V[j*p:(j+1)*p, :]
                   # W[:, i] is W[::n, i] ? No, W is (p*n, R). Index k*n+i.
                   # W[k, i] corresponds to row k*n+i in flat W.
                   # So W[:, i] in logical (p, n, R) is W.reshape(p, n, R)[:, i, :]
                   A = V.reshape(m, p, rank)[j] * W.reshape(p, n, rank)[:, i, :]
                   # A is (p, rank)
                   b = np.ones(p)
                   try:
                       u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                       U_new[i * m + j, :] = u_sol
                   except:
                       U_new[i * m + j, :] = U[i * m + j, :]
           
           # Update V
           V_new = np.zeros_like(V)
           for j in range(m):
               for k in range(p):
                   # A[i, r] = U[i, j, r] * W[k, i, r]
                   # U[i, j] is U.reshape(n, m, R)[i, j, :]
                   # W[k, i] is W.reshape(p, n, R)[k, i, :]
                   A = U.reshape(n, m, rank)[:, j, :] * W.reshape(p, n, rank)[k, :, :]
                   # A is (n, rank)
                   b = np.ones(n)
                   try:
                       v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                       V_new[j * p + k, :] = v_sol
                   except:
                       V_new[j * p + k, :] = V[j * p + k, :]
           
           # Update W
           W_new = np.zeros_like(W)
           for k in range(p):
               for i in range(n):
                   # A[j, r] = U[i, j, r] * V[j, k, r]
                   # U[i, j] is U.reshape(n, m, R)[i, j, :]
                   # V[j, k] is V.reshape(m, p, R)[j, k, :]
                   A = U.reshape(n, m, rank)[i, :, :] * V.reshape(m, p, rank)[:, k, :]
                   # A is (m, rank)
                   b = np.ones(m)
                   try:
                       w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                       W_new[k * n + i, :] = w_sol
                   except:
                       W_new[k * n + i, :] = W[k * n + i, :]
           
           U, V, W = U_new, V_new, W_new
           
           # Check error
           # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
           # Use einsum
           T = np.einsum('ijr,jkr,kir->ijk', U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank))
           error = np.sum((T - 1.0)**2)
           if error < tol:
               break
               
       T = np.einsum('ijr,jkr,kir->ijk', U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank))
       error = np.sum((T - 1.0)**2)
       if error < 1e-8:
           return U, V, W
       return None
   