    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        # Vectorized verification
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # This is a 3-tensor contraction.
        # T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        # However, 'kir' means k, i, r. W is indexed by k, i.
        # W_reshaped is (p, n, r). Index k, i, r.
        # So 'kir' is correct for W.
        # U is 'ijr'. V is 'jkr'.
        # Result T[i, j, k]
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            # Check if all elements are 1
            if not np.allclose(T, 1.0, atol=tol):
                return False
            return True
        except Exception:
            return False
