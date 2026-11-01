<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time

def _find_rank5_2x3():
    """Find a rank-5 decomposition for 2x2 * 2x3 -> 2x3 matrix multiplication.
    Returns (U, V, W) each of shape (4,5), (6,5), (6,5) respectively.
    U indices: i*2+j for i in 0..1, j in 0..1 (A elements)
    V indices: j*3+k for j in 0..1, k in 0..2 (B elements)
    W indices: k*2+i for k in 0..2, i in 0..1 (C elements)
    """
    n, m, p = 2, 2, 3
    # Build the multiplication tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in range
    # Tensor shape: (n*m, m*p, p*n) = (4, 6, 6)
    # T[a_idx, b_idx, c_idx] where a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    rank = 5
    # Use randomized SVD / iterative approach to find rank-5 decomposition
    # Initialize with random matrices
    best = None
    np.random.seed(42)
    for trial in range(100):
        U_init = np.random.randn(n*m, rank)
        V_init = np.random.randn(m*p, rank)
        W_init = np.random.randn(p*n, rank)
        
        # Iteratively refine using ALS-like approach
        for it in range(200):
            # Fix V, W, solve for U
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed V,W: T[:,b,c] = U @ (V[b,:]*W[c,:])
            # Reshape: for each r, U[:,r] should satisfy sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is a linear system: U @ (V[b,:] outer W[c,:]) = T[:,b,c]
            # Better: vec form. T reshaped as (n*m, m*p*p*n) = (U ⊗ I) * something
            # Simpler: for each r, the contribution is U[:,r] outer (V[:,r] * W[:,r])
            # So T = sum_r U[:,r] outer (V[:,r] * W[:,r])
            # Flatten: T_flat = sum_r U[:,r] * (V[:,r] * W[:,r]) reshaped appropriately
            
            # Use the fact that T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # For fixed V,W: for each a, T[a,:,:] = U[a,:] @ M where M[r,(b,c)] = V[b,r]*W[c,r]
            # M shape: (rank, m*p * p*n)
            M = np.zeros((rank, m*p * p*n))
            for b in range(m*p):
                for c in range(p*n):
                    for r in range(rank):
                        M[r, b * p*n + c] = V_init[b, r] * W_init[c, r]
            # T_flat shape: (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, m*p * p*n)
            # Solve U_init @ M^T = T_flat => U_init = T_flat @ M (pinv)
            U_new = T_flat @ np.linalg.pinv(M.T)
            
            # Fix U, W, solve for V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed U,W: T[a,b,c] = V[b,:] @ (U[a,:] * W[c,:])
            M2 = np.zeros((rank, n*m * p*n))
            for a in range(n*m):
                for c in range(p*n):
                    for r in range(rank):
                        M2[r, a * p*n + c] = U_new[a, r] * W_init[c, r]
            T_flat2 = T.reshape(m*p, n*m * p*n).T  # (p*n, m*p) -> need (n*m*p*n, m*p)
            # T[a,b,c] indexed as (b, a*p*n+c)
            T_flat2 = np.zeros((m*p, n*m * p*n))
            for b in range(m*p):
                for a in range(n*m):
                    for c in range(p*n):
                        T_flat2[b, a * p*n + c] = T[a, b, c]
            V_new = T_flat2 @ np.linalg.pinv(M2.T)
            
            # Fix U, V, solve for W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed U,V: T[a,b,c] = W[c,:] @ (U[a,:] * V[b,:])
            M3 = np.zeros((rank, n*m * m*p))
            for a in range(n*m):
                for b in range(m*p):
                    for r in range(rank):
                        M3[r, a * m*p + b] = U_new[a, r] * V_new[b, r]
            T_flat3 = np.zeros((p*n, n*m * m*p))
            for c in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                    T_flat3[c, a * m*p + b] = T[a, b, c]
            W_new = T_flat3 @ np.linalg.pinv(M3.T)
            
            U_init, V_init, W_init = U_new, V_new, W_new
        
        # Check error
        error = 0
        for r in range(rank):
            contrib = np.outer(U_init[:, r], np.outer(V_init[:, r], W_init[:, r]).reshape(m*p, p*n))
            error += np.abs(contrib).sum()
        # Better: reconstruct and compare
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            recon += np.outer(U_init[:, r], np.outer(V_init[:, r], W_init[:, r]).reshape(m*p, p*n))
        err = np.abs(recon - T).max()
        if best is None or err < best[0]:
            best = (err, U_init.copy(), V_init.copy(), W_init.copy())
    
    err, U, V, W = best
    # Round to nearest half-integer
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify rounded version
    recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        recon += np.outer(U_r[:, r], np.outer(V_r[:, r], W_r[:, r]).reshape(m*p, p*n))
    err_r = np.abs(recon - T).max()
    
    if err_r < 1e-9:
        return U_r, V_r, W_r
    else:
        # Fall back to schoolbook for this block
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return U_s, V_s, W_s

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Split 12 columns into 4 blocks of 3
    block_size = 3
    num_blocks = p // block_size  # 4
    
    # Find rank-5 decomposition for 2x2 * 2x3
    U_block, V_block, W_block = _find_rank5_2x3()
    block_rank = U_block.shape[1]
    
    total_rank = num_blocks * block_rank
    
    # Build full decomposition
    # A is 2x2 (4 elements), B is 2x12 (24 elements), C is 2x12 (24 elements)
    # For block b (0..3), B columns 3b..3b+2, C columns 3b..3b+2
    # U: shape (4, total_rank), V: shape (24, total_rank), W: shape (24, total_rank)
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for b in range(num_blocks):
        col_offset = b * block_rank
        for r in range(block_rank):
            # U: same for all blocks (A is shared)
            U_full[:, col_offset + r] = U_block[:, r]
            
            # V: B element at row j, col 3b+k maps to index j*p + (3b+k)
            # V_block has index j*3+k for j in 0..1, k in 0..2
            for j in range(m):
                for k in range(block_size):
                    v_idx_block = j * block_size + k
                    v_idx_full = j * p + (b * block_size + k)
                    V_full[v_idx_full, col_offset + r] = V_block[v_idx_block, r]
            
            # W: C element at row i, col 3b+k maps to index (3b+k)*n + i
            # W_block has index k*2+i for k in 0..2, i in 0..1
            for k in range(block_size):
                for i in range(n):
                    w_idx_block = k * n + i
                    w_idx_full = (b * block_size + k) * n + i
                    W_full[w_idx_full, col_offset + r] = W_block[w_idx_block, r]
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    result = {
        "U": to_list(U_full),
        "V": to_list(V_full),
        "W": to_list(W_full)
    }
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return schoolbook as fallback
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I notice there's a syntax error in my code (missing colon in a for loop). Let me fix that:

