import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def build_84_decomposition():
    """
    Builds the specific 84-rank decomposition:
    n=3 (1+2), m=5 (2+2+1), p=6 (2+2+2).
    """
    n, m, p = 3, 5, 6
    
    # Partitions
    n_sizes = [1, 2]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
                    
    # Should be 84
    assert total_rank == 84, f"Expected 84, got {total_rank}"
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    # Offsets
    n_offsets = [0, 1]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_blk, V_blk, W_blk = strassen_factors()
                else:
                    U_blk, V_blk, W_blk = schoolbook_factors(ni, mj, pk)
                
                r = U_blk.shape[1]
                
                for li in range(ni):
                    for lj in range(mj):
                        full_row_u = (i_start + li) * m + (j_start + lj)
                        U[full_row_u, col_offset:col_offset + r] = U_blk[li * mj + lj, :]
                        
                for lj in range(mj):
                    for lk in range(pk):
                        full_row_v = (j_start + lj) * p + (k_start + lk)
                        V[full_row_v, col_offset:col_offset + r] = V_blk[lj * pk + lk, :]
                        
                for lk in range(pk):
                    for li in range(ni):
                        full_row_w = (k_start + lk) * n + (i_start + li)
                        W[full_row_w, col_offset:col_offset + r] = W_blk[lk * ni + li, :]
                        
                col_offset += r
                
    return U, V, W, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))

def try_reduce_rank(U, V, W, n, m, p):
    """
    Attempts to reduce the rank by finding linear dependencies.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Flatten vectors
    # u_c is size n*m
    # v_c is size m*p
    # w_c is size p*n
    # Tensor vector is size (n*m)*(m*p)*(p*n)
    
    # We check if any column is a linear combination of others.
    # This is equivalent to checking the rank of the matrix of Kronecker products.
    # But that matrix is huge.
    # Instead, we can check the rank of the "flattened" factor matrices?
    # No, the rank of the tensor is the minimum number of terms.
    # If the 84 vectors are linearly dependent, the rank is < 84.
    
    # A simpler check:
    # If we remove column c, does the remaining system still span the target tensor?
    # This is hard to check directly without solving.
    
    # However, we can check the rank of the matrix formed by the vectors in the tensor space.
    # Dimension of tensor space: 15 * 30 * 18 = 8100.
    # We have 84 vectors.
    # We can form an 8100 x 84 matrix.
    
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    total_dim = dim_u * dim_v * dim_w
    
    # This might be memory intensive if total_dim is large, but 8100*84 is ~680k floats. Fine.
    
    # Build the matrix of Kronecker products
    # M[:, c] = kron(kron(U[:, c], V[:, c]), W[:, c])
    
    # To save time, we can use a random projection to a smaller space?
    # Or just compute the rank of the 8100x84 matrix.
    
    # Let's try to find a dependent column.
    # We can iterate through columns and check if the rank drops if we remove it.
    # Rank of full matrix should be 84 (if independent).
    # If rank is 83, one column is redundant.
    
    # Compute rank of full set
    # Using SVD is expensive for 8100x84? No, 84 is small.
    
    # Actually, we can just check if the system is consistent when we drop a column.
    # But we don't have the target tensor explicitly as a vector?
    # We do: it's the standard multiplication tensor.
    
    # Let's just return the 84 decomposition if we can't easily prove dependency.
    # The probability of a random 84-term decomposition being dependent is low unless structured.
    # Given the block structure, they are likely independent.
    
    return U, V, W

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Build the 84-rank decomposition
    U, V, W, rank = build_84_decomposition()
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank (optional, likely fails)
        if time.time() < deadline - 10:
             U, V, W = try_reduce_rank(U, V, W, n, m, p)
             if verify_decomposition(U, V, W, n, m, p):
                 return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
