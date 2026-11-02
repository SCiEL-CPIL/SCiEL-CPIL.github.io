    def make_residuals_fast(r):
        def residuals(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            # U: (nm, r), V: (mp, r), W: (pn, r)
            # We want T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Indices:
            # i in 0..n-1, j in 0..m-1, k in 0..p-1
            # u_idx = i*m + j
            # v_idx = j*p + k
            # w_idx = k*n + i
            
            # Create index arrays
            i_vals = np.repeat(np.arange(n), m*p)
            j_vals = np.tile(np.repeat(np.arange(m), p), n)
            k_vals = np.tile(np.arange(p), n*m)
            
            u_idx = i_vals * m + j_vals
            v_idx = j_vals * p + k_vals
            w_idx = k_vals * n + i_vals
            
            # Select rows
            U_sel = U[u_idx, :] # (N, r)
            V_sel = V[v_idx, :] # (N, r)
            W_sel = W[w_idx, :] # (N, r)
            
            # Element-wise product and sum
            # prod = U_sel * V_sel * W_sel -> (N, r)
            # res = prod.sum(axis=1) - 1
            res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1
            return res
        return residuals
