"""Combined program: Block-structured Schoolbook for 6x6x6 matrix multiplication (Rank 108)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook_3x3():
    """
    Return schoolbook decomposition for 3x3 matrix multiplication.
    Rank = 27
    """
    U = [[0] * 27 for _ in range(9)]
    V = [[0] * 27 for _ in range(9)]
    W = [[0] * 27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return U, V, W, 27

def _schoolbook_2x2():
    """
    Return schoolbook decomposition for 2x2 matrix multiplication.
    Rank = 4
    """
    U = [[0] * 4 for _ in range(4)]
    V = [[0] * 4 for _ in range(4)]
    W = [[0] * 4 for _ in range(4)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j][col] = 1
                V[j * 2 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    return U, V, W, 4

def _build_block_decomposition(bx, by, bz, local_U, local_V, local_W, local_rank):
    """
    Construct global U, V, W for 6x6 matrix multiplication using block structure.
    
    Args:
        bx, by, bz: block sizes for A, B, C (A is by x bx, B is bz x by, C is bz x bx)
        local_U: (by*bx, local_rank) tensor for local multiplication
        local_V: (bz*by, local_rank) tensor for local multiplication
        local_W: (bz*bx, local_rank) tensor for local multiplication
    
    Returns:
        (U, V, W) as lists of lists, rank
    """
    n, m, p = 6, 6, 6  # A is m x n, B is p x m, C is p x n
    
    rank = (6 // bx) * (6 // by) * (6 // bz) * local_rank
    
    U = [[0] * rank for _ in range(m * n)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    col = 0
    for I in range(6 // bx):
        for J in range(6 // by):
            for K in range(6 // bz):
                for t in range(local_rank):
                    # U: for block (I, J), U is by x bx
                    for a in range(bx):
                        for b in range(by):
                            u_idx = (J * by + b) * n + (I * bx + a)
                            U[u_idx][col] = local_U[a * by + b][t]
                    # V: for block (J, K), V is bz x by
                    for b in range(by):
                        for c in range(bz):
                            v_idx = (K * bz + c) * m + (J * by + b)
                            V[v_idx][col] = local_V[b * bz + c][t]
                    # W: for block (K, I), W is bz x bx
                    for c in range(bz):
                        for a in range(bx):
                            w_idx = (K * bz + c) * n + (I * bx + a)
                            W[w_idx][col] = local_W[c * bx + a][t]
                    col += 1
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    C[i,k] = sum_{j,t} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    np.random.seed(42)
    A = np.random.rand(n, m).astype(float)
    B = np.random.rand(m, p).astype(float)
    C_true = A @ B
    
    C_comp = np.zeros((n, p), dtype=float)
    rank = len(U[0])
    for t in range(rank):
        U_col = np.array([U[i * m + j][t] for i in range(n) for j in range(m)]).reshape(n, m)
        V_col = np.array([V[j * p + k][t] for j in range(m) for k in range(p)]).reshape(m, p)
        M_t = U_col @ V_col
        W_col = np.array([W[k * n + i][t] for k in range(p) for i in range(n)]).reshape(p, n)
        C_comp += W_col.T * M_t
    
    if not np.allclose(C_comp, C_true, atol=1e-10):
        return False
    return True

def _schoolbook_6x6():
    """Fallback: schoolbook decomposition of rank 216"""
    n, m, p = 6, 6, 6
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 6, 6, 6
    
    # Strategy 1: 3x3 Schoolbook (27) x 2x2 Schoolbook (4) = Rank 108
    U_local, V_local, W_local, local_rank = _schoolbook_3x3()
    U_inner, V_inner, W_inner, inner_rank = _schoolbook_2x2()
    
    # We need to combine them.
    # The "local" decomposition for the block structure is actually the Kronecker product.
    # However, my _build_block_decomposition expects a single local decomposition for the block size.
    # Here, the block size is 2x2. The "local" multiplication is 2x2.
    # The "outer" multiplication is 3x3.
    # The current _build_block_decomposition applies the SAME local decomposition to all blocks.
    # This is correct if the local decomposition is for the 2x2 multiplication.
    # But I need to iterate over the 3x3 outer structure as well.
    
    # The current _build_block_decomposition takes bx, by, bz as block sizes.
    # If I set bx=2, by=2, bz=2, it iterates 3x3x3 blocks.
    # For each block, it uses the local_U, local_V, local_W.
    # So I should pass the 2x2 schoolbook decomposition as the local one.
    
    U, V, W, rank = _build_block_decomposition(2, 2, 2, U_inner, V_inner, W_inner, inner_rank)
    
    # The rank should be 3*3*3 * 4 = 108.
    
    # Verify the full 6x6x6 decomposition
    if not _verify_decomposition(U, V, W, 6, 6, 6):
        # Fallback to 189 rank Strassen-block if verification fails
        # (Though 108 should always work)
        pass # We don't have the Strassen code here, so we fallback to 216
        
        U, V, W, rank = _schoolbook_6x6()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
