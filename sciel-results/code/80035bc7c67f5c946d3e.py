"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i, j, k] = 1 for all i, j, k
    # T is a tensor of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Reshape T to a matrix T_mat of shape (n*m, p) by flattening the first two dimensions
    # T_mat[i*m+j, k] = T[i, j, k]
    T_mat = T.reshape(n * m, p)
    
    # The rank of the matrix multiplication is at least the matrix rank of T_mat
    # Compute the rank of T_mat using SVD
    U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
    
    # The rank is the number of non-zero singular values
    tol = 1e-10
    rank_svd = np.sum(s > tol)
    
    # The tensor rank is at least the matrix rank, but could be higher
    # However, for matrix multiplication tensors, the rank is often close to the matrix rank
    # Let's try to find a decomposition of rank equal to rank_svd
    
    # If rank_svd is small, we can try to construct a decomposition
    # For now, let's try a few strategies
    
    # Strategy 1: Use the SVD to get a low-rank decomposition
    # T_mat = U_svd @ diag(s) @ Vt
    # We can write T_mat = U_svd_rank @ diag(s_rank) @ Vt_rank
    # where U_svd_rank = U_svd[:, :rank_svd] * s[:rank_svd]
    # and Vt_rank = Vt[:rank_svd, :]
    
    # But we need a tensor decomposition, not a matrix decomposition
    # The tensor decomposition requires that the decomposition respects the structure
    
    # For matrix multiplication, a rank-r decomposition is a set of r rank-1 tensors
    # T = sum_{c=1}^r T_1^{(c)} \otimes T_2^{(c)} \otimes T_3^{(c)}
    
    # Each rank-1 tensor T_1^{(c)} \otimes T_2^{(c)} \otimes T_3^{(c)} corresponds to
    # a term in the decomposition where:
    # U[i*m+j, c] = T_1^{(c)}[i] * T_2^{(c)}[j]
    # V[j*p+k, c] = T_2^{(c)}[j] * T_3^{(c)}[k]
    # W[k*n+i, c] = T_3^{(c)}[k] * T_1^{(c)}[i]
    
    # So we need to find r triples (a_c, b_c, c_c) where a_c is a vector of length n,
    # b_c is a vector of length m, c_c is a vector of length p, such that:
    # T[i, j, k] = sum_{c=1}^r a_c[i] * b_c[j] * c_c[k]
    
    # This is exactly the problem of finding a rank-r decomposition of the tensor T
    # We can try to use the SVD as a starting point
    
    # Let's try a simple approach: use the SVD to get a rank-r decomposition of T_mat
    # and then try to "lift" it to a tensor decomposition
    
    # Actually, for matrix multiplication tensors, the rank is known to be:
    # M(2, 5, 5) is not known exactly, but upper bounds are known
    
    # Let's try a few candidate ranks and see if we can find a valid decomposition
    # We'll use a greedy approach or a random search
    
    # For now, let's return the schoolbook decomposition as a fallback
    # and try to improve it if we have time
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
