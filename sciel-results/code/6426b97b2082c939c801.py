<<<<<<< SEARCH
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[:,c] outer W[:,c] * V[:,c] equals matmul tensor.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    Tensor T[i,j,k] = sum_ab A[i,a] B[b,k] * delta_{j,a} * delta_{b,j} ... no.
    Actually T[i,j,k] = A[i,j] * B[j,k].
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    import numpy as np
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = np.array(U[:, c])
        vc = np.array(V[:, c])
        wc = np.array(W[:, c])
        uc = uc.reshape(n, m)
        vc = vc.reshape(m, p)
        wc = wc.reshape(p, n)
        # contribution to T[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # This is a rank-1 tensor in (i,j,k) space.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
    
    # Target tensor: T[i,j,k] = 1 if we think of A[i,j]=1, B[j,k]=1? No.
    # The bilinear map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor T_matmul has components T[i,j,k] = 1 if the term A[i,j]B[j,k] appears.
    # Actually, the standard tensor for matrix multiplication of n x m by m x p has:
    # T[i, j, k] = 1 for all i in [n], j in [m], k in [p].
    # Because C[i,k] = sum_j A[i,j] B[j,k]. Each term is A[i,j] B[j,k].
    # So the tensor is the all-ones tensor? No, that would be rank 1 if all ones.
    # Wait. The tensor is indexed by (i, j, k). The value is 1 for all (i,j,k).
    # Let's check: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # If this equals 1 for all i,j,k, then it's the all-ones tensor.
    # But the schoolbook decomposition has rank n*m*p.
    # Let's check schoolbook:
    # For each (i,j,k), there is a term c(i,j,k) where U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # All other entries for this c are 0.
    # So sum_c ... = 1 for each (i,j,k).
    # Yes, the matrix multiplication tensor for n x m by m x p is the all-ones tensor of size n x m x p.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i,j,k] != 1.0:
                    return False
    return True
=======
def _verify_decomposition(U, V, W, n, m, p):
    """Verify decomposition using NumPy for speed and correctness."""
    import numpy as np
    if not U:
        return True
    rank = len(U[0])
    U_arr = np.array(U).T  # Shape (rank, n*m)
    V_arr = np.array(V).T  # Shape (rank, m*p)
    W_arr = np.array(W).T  # Shape (rank, p*n)
    
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(rank, n, m)
    V_3d = V_arr.reshape(rank, m, p)
    W_3d = W_arr.reshape(rank, p, n)
    
    # Compute sum over rank dimension of outer products
    # T[i,j,k] = sum_c U[c,i,j] * V[c,j,k] * W[c,k,i]
    # We can compute this efficiently.
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_3d[c]
        vc = V_3d[c]
        wc = W_3d[c]
        # Contribution is uc[i,j] * vc[j,k] * wc[k,i]
        # This is not a simple element-wise multiplication because indices differ.
        # We need to sum over c of (uc[i,j] * vc[j,k] * wc[k,i])
        # Let's iterate over i, j, k or use einsum.
        pass # The loop below is safer for clarity in this context, but let's try einsum if possible.
        # Actually, for small dimensions, a simple loop or vectorized op is fine.
        # Let's use a vectorized approach:
        # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # This is a bit tricky to vectorize without creating large intermediates.
        # Given n,m,p <= 6, rank ~ 63, the loop over c is fast.
        # Inside the loop, we can do:
        # temp = np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        # T += temp
        T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    
    # The target tensor is all ones
    if not np.allclose(T, np.ones((n, m, p))):
        return False
    return True
>>>>>>> REPLACE

