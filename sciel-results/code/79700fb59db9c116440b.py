import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    # Build the matrix multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Flatten T into a matrix of shape (n*p, m)
    # M[i*p+k, j] = T[i, j, k]
    M = np.zeros((n * p, m), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * p + k, j] = T[i, j, k]
    
    # Compute the rank of M over the reals
    rank = int(np.linalg.matrix_rank(M))
    
    # Compute the SVD to find a basis for the row space
    U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    
    # The row space basis is given by the rows of Vt_svd corresponding to non-zero singular values
    # We use the first 'rank' rows of Vt_svd as the basis vectors
    basis = Vt_svd[:rank, :]
    
    # Now we need to express each column of M as a linear combination of the basis vectors
    # M = basis^T @ X for some matrix X of shape (rank, m)
    # Actually, M[i, j] = sum_c basis[c, j] * X[c, i] ? No.
    
    # Let's be precise.
    # The rows of M are in the span of the basis vectors (rows of Vt_svd).
    # So M = Y @ basis, where Y is (n*p, rank) and basis is (rank, m).
    # Since basis has orthonormal rows (from SVD), Y = M @ basis^T.
    Y = M @ basis.T
    
    # Now, for a fixed j, the column M[:, j] is a vector in R^(n*p).
    # We can write M[:, j] = Y[:, j] @ basis.
    # Wait, Y is (n*p, rank). So M[:, j] = sum_c Y[c, j] * basis[c, :]? No.
    # M[:, j] is a column vector. basis is (rank, m).
    # The relationship is M = Y @ basis.
    # So M[:, j] = Y @ basis[:, j].
    # This means that for each j, the vector M[:, j] is a linear combination of the columns of Y?
    # No, M[:, j] = sum_c Y[c, j] * basis[c, :] ... this is not a vector in R^(n*p).
    
    # Let's rethink.
    # We want to decompose the tensor T[i, j, k] into sum_c U[i, c] * V[j, c] * W[k, c] ?
    # The interface is:
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]
    # W: (p*n, rank) -> W[k*n+i, c]
    
    # The condition is:
    # C[i, k] = sum_j A[i, j] B[j, k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] A[i',j']) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    
    # This is a rank-r decomposition of the 3-tensor T[i, j, k].
    # T[i, j, k] = sum_c U_c[i, j] V_c[j, k] W_c[k, i] ? No, the indices don't match up that way.
    
    # Let's look at the schoolbook again.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Then sum_c W[k*n+i, c] U[i*m+j, c] V[j*p+k, c] = 1 if there exists c such that all three are 1.
    # In schoolbook, each c corresponds to a unique (i, j, k).
    # So for a given (i, k), the sum is over all c such that W[k*n+i, c]=1, U[i*m+j_c, c]=1, V[j_c*p+k, c]=1.
    # This forces j_c = j. So the sum is 1 for each j, and 0 otherwise.
    # Thus C[i, k] = sum_j A[i, j] B[j, k].
    
    # The general condition is that the tensor T[i, j, k] can be written as:
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # This is exactly the definition of the tensor rank.
    # We can find a minimal rank decomposition by finding the rank of the matrix M[i*p+k, j] = T[i, j, k].
    # If rank(M) = r, then there exist vectors u_c in R^(n*p) and v_c in R^m such that M = sum_c u_c v_c^T.
    # u_c[i*p+k] = sum_c' U[i*m+j', c'] ... this is getting complicated.
    
    # Let's use the SVD result.
    # M = Y @ basis, where Y is (n*p, rank) and basis is (rank, m).
    # M[i*p+k, j] = sum_c Y[i*p+k, c] * basis[c, j].
    
    # We want T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # Note that T[i, j, k] = M[i*p+k, j].
    # So M[i*p+k, j] = sum_c Y[i*p+k, c] * basis[c, j].
    
    # Can we match this to the form sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]?
    # If we set W[k*n+i, c] = Y[i*p+k, c] (note the index swap: Y is indexed by i*p+k, W by k*n+i),
    # and V[j*p+k, c] = 1 (for all j, k), and U[i*m+j, c] = basis[c, j] (for all i),
    # then sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = sum_c basis[c, j] * 1 * Y[i*p+k, c] = M[i*p+k, j] = T[i, j, k].
    
    # Let's verify this construction.
    # U has shape (n*m, rank). U[i*m+j, c] = basis[c, j].
    # V has shape (m*p, rank). V[j*p+k, c] = 1.
    # W has shape (p*n, rank). W[k*n+i, c] = Y[i*p+k, c].
    
    # Check: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # = sum_c basis[c, j] * 1 * Y[i*p+k, c]
    # = sum_c Y[i*p+k, c] * basis[c, j]
    # = M[i*p+k, j]
    # = T[i, j, k].
    
    # This is a valid decomposition!
    # The rank is rank(M).
    
    # Let's construct the matrices.
    U_mat = np.zeros((n * m, rank), dtype=float)
    V_mat = np.zeros((m * p, rank), dtype=float)
    W_mat = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                U_mat[i * m + j, c] = basis[c, j]
        
        for k in range(p):
            for c in range(rank):
                W_mat[k * n + i, c] = Y[i * p + k, c]
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V_mat[j * p + k, c] = 1.0
    
    # Convert to lists as required by the interface
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
