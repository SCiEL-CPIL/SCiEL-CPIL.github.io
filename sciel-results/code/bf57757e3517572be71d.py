import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_decomposition():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U row: i*2 + j (A_ij)
    V row: j*2 + k (B_jk)
    W row: k*2 + i (C_ik) -- Note: Task says W[k*n+i, r]
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U entries (A coefficients)
    # Row indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # Wait, W indices are k*n+i.
    # C11 (k=0,i=0) -> W[0]
    # C12 (k=1,i=0) -> W[2]
    # C21 (k=0,i=1) -> W[1]
    # C22 (k=1,i=1) -> W[3]
    
    # Let's re-verify W index mapping: W[k*n+i, r].
    # n=2.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C21
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    W[1, 3] = 1  # C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[2, 4] = 1  # C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W

def _get_simple_decomposition(n, m, p):
    """
    Returns U, V, W for n x m x p multiplication where rank is 1.
    This is valid if any of n,m,p is 1.
    U: (n*m, 1)
    V: (m*p, 1)
    W: (p*n, 1)
    """
    U = np.ones((n*m, 1))
    V = np.ones((m*p, 1))
    W = np.ones((p*n, 1))
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Block sizes
    block_sizes = [2, 2, 1]
    
    # Get base decompositions
    U_222, V_222, W_222 = _get_strassen_decomposition()
    
    # We will accumulate U, V, W
    # Total rank will be sum of ranks of blocks
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Pre-calculate simple decompositions for mixed sizes
    # We only need (1,1,1), (1,2,1), (2,1,1), (1,1,2), (1,2,2), (2,1,2), (2,2,1)
    # But since rank is 1 for all these, we can just generate them on the fly or cache.
    
    # Iterate over blocks
    # i: block index for rows of A (and cols of C)
    # j: block index for cols of A (and rows of B)
    # k: block index for cols of B (and rows of C)
    
    for i in range(3):
        ni = block_sizes[i]
        for j in range(3):
            mj = block_sizes[j]
            for k in range(3):
                pk = block_sizes[k]
                
                # Determine rank and get decomposition
                if ni == 2 and mj == 2 and pk == 2:
                    rank = 7
                    U_sub = U_222
                    V_sub = V_222
                    W_sub = W_222
                else:
                    rank = 1
                    U_sub, V_sub, W_sub = _get_simple_decomposition(ni, mj, pk)
                
                # Calculate row offsets in global matrices
                # U rows: i_global * m + j_global
                # The block (i,j) covers rows i_start to i_end in A, cols j_start to j_end in A.
                # A is n x m.
                # Row offset in U: sum of sizes of previous i-blocks * m + sum of sizes of previous j-blocks
                # Wait, U index is i*m + j.
                # i is the block index. The global row index for A is I.
                # The block i corresponds to global rows I_start to I_end.
                # I_start = sum(block_sizes[0:i])
                # The local row index in U_sub is l_i * mj + l_j.
                # The global row index is I_start + l_i, J_start + l_j.
                # Global U index = (I_start + l_i) * m + (J_start + l_j).
                
                I_start = sum(block_sizes[0:i])
                J_start = sum(block_sizes[0:j])
                K_start = sum(block_sizes[0:k])
                
                # Construct full row index mappings
                # U_sub has shape (ni*mj, rank)
                # We need to create a full U_block of shape (n*m, rank) with zeros, and fill the block.
                # Or we can just append to lists if we process column by column?
                # No, U, V, W are matrices. We need to align columns.
                # Since we are adding rank terms, we can just build the full matrices.
                
                # To be efficient, we can pre-allocate the full U, V, W matrices with total rank.
                # But we don't know total rank yet.
                # Let's collect the sub-matrices and their offsets.
                
                U_block = np.zeros((n*m, rank))
                V_block = np.zeros((m*p, rank))
                W_block = np.zeros((p*n, rank))
                
                for r in range(rank):
                    # Fill U_block
                    for u_idx in range(ni * mj):
                        l_i = u_idx // mj
                        l_j = u_idx % mj
                        g_row_u = (I_start + l_i) * m + (J_start + l_j)
                        U_block[g_row_u, r] = U_sub[u_idx, r]
                        
                    # Fill V_block
                    for v_idx in range(mj * pk):
                        l_j = v_idx // pk
                        l_k = v_idx % pk
                        g_row_v = (J_start + l_j) * p + (K_start + l_k)
                        V_block[g_row_v, r] = V_sub[v_idx, r]
                        
                    # Fill W_block
                    for w_idx in range(pk * ni):
                        l_k = w_idx // ni
                        l_i = w_idx % ni
                        g_row_w = (K_start + l_k) * n + (I_start + l_i)
                        W_block[g_row_w, r] = W_sub[w_idx, r]
                        
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)
                total_rank += rank
                
    # Concatenate
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    # Convert to lists as required by interface?
    # The interface says: {'U', 'V', 'W'}: factor matrices ... with integer/half-integer entries
    # The seed program returned lists of lists.
    # Let's check if numpy arrays are accepted.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually numpy arrays are fine, but to be safe, we can convert to lists if the evaluator expects lists.
    # The seed used lists. I will stick to lists to be safe, or check if the evaluator handles arrays.
    # Given the seed used lists, I will return lists.
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
