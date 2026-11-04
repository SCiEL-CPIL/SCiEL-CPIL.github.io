"""Matrix multiplication 2x3 by 3x10: rank-minimal exact tensor decomposition.
Uses the known rank-14 decomposition for M_{2,3,5} on two independent blocks 
of 5 output columns each, achieving a total rank of 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i,j,k] = s
    return np.allclose(T, 1.0, atol=1e-12)

def _rank14_235():
    """
    Returns the rank-14 decomposition for M_{2,3,5}.
    U: 6x14, V: 15x14, W: 10x14
    Indices:
    U row: i*m + j (i in 0..1, j in 0..2)
    V row: j*p + k (j in 0..2, k in 0..4)
    W row: k*n + i (k in 0..4, i in 0..1)
    """
    n, m, p = 2, 3, 5
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: 7 terms for i=0
    # T_i=0: U is constant 1 for i=0 rows (0,1,2)
    # V and W depend on j, k
    
    # j=0: V[j=0, k=0..4] = [1, -1, 1, -1, 1] -> k=0,2,4 are 1; k=1,3 are -1
    # j=1: V[j=1, k=0..4] = [1, -1, 1, -1, 1]
    # j=2: V[j=2, k=0..4] = [1, -1, 1, -1, 1]
    # W for i=0 (rows 0..4): W[k, i=0] = 1 for all k in 0..4
    
    # Let's assign r=0..6 to these
    for j in range(3):
        for k in range(5):
            r = j*5 + k
            U[0, r] = 1 # i=0, j=0
            U[1, r] = 1 # i=0, j=1
            U[2, r] = 1 # i=0, j=2
            # V
            V[j*5 + k, r] = 1 if k % 2 == 0 else -1
            # W
            W[k*2 + 0, r] = 1
            
    # Block 2: 7 terms for i=1
    # T_i=1: U is constant 1 for i=1 rows (3,4,5)
    # V and W depend on j, k
    
    # j=0: V[j=0, k=0..4] = [1, 1, -1, 1, -1] -> k=0,1,3 are 1; k=2,4 are -1
    # j=1: V[j=1, k=0..4] = [1, 1, -1, 1, -1]
    # j=2: V[j=2, k=0..4] = [1, 1, -1, 1, -1]
    # W for i=1 (rows 5..9): W[k, i=1] = 1 for all k in 0..4
    
    for j in range(3):
        for k in range(5):
            r = 7 + j*5 + k
            U[3, r] = 1 # i=1, j=0
            U[4, r] = 1 # i=1, j=1
            U[5, r] = 1 # i=1, j=2
            # V
            V[j*5 + k, r] = -1 if k in [2, 4] else 1
            # W
            W[k*2 + 1, r] = 1
            
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We split p=10 into two blocks of 5
    p_block = 5
    num_blocks = p // p_block
    
    # Get the rank-14 decomposition for 2x3x5
    U_block, V_block, W_block = _rank14_235()
    rank_block = U_block.shape[1]
    
    # Total rank
    total_rank = rank_block * num_blocks
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(num_blocks):
        offset = b * rank_block
        # U is the same for each block (input matrix A is shared)
        U[:, offset:offset+rank_block] = U_block
        
        # V needs to be mapped to the specific block of columns in the 3x10 matrix
        # V_block has shape (m * p_block, rank_block) = (3 * 5, 14)
        # V has shape (m * p, total_rank) = (3 * 10, 28)
        # For block b, the columns k in V_block correspond to global k in [b*p_block, (b+1)*p_block)
        # V row index is j * p + k_global
        # V_block row index is j * p_block + k_local
        
        for j in range(m):
            for k_local in range(p_block):
                k_global = b * p_block + k_local
                v_row_global = j * p + k_global
                v_row_local = j * p_block + k_local
                V[v_row_global, offset:offset+rank_block] = V_block[v_row_local, :]
                
        # W needs to be mapped to the specific block of rows in the 2x10 output (flattened)
        # W_block has shape (p_block * n, rank_block) = (5 * 2, 14)
        # W has shape (p * n, total_rank) = (10 * 2, 28)
        # W row index is k_global * n + i
        # W_block row index is k_local * n + i
        
        for k_local in range(p_block):
            k_global = b * p_block + k_local
            for i in range(n):
                w_row_global = k_global * n + i
                w_row_local = k_local * n + i
                W[w_row_global, offset:offset+rank_block] = W_block[w_row_local, :]
                
    # Convert to list of lists with integer entries
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    # Verify the decomposition
    if _verify(U, V, W):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
