"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # C00 (k=0, i=0)
    W[3][0] = 1  # C11 (k=1, i=1)
    
    # M2: (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[1][1] = 1  # C10 (k=0, i=1)
    W[3][1] = -1 # C11 (k=1, i=1)
    
    # M3: a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[2][2] = 1  # C01 (k=1, i=0)
    W[3][2] = 1  # C11 (k=1, i=1)
    
    # M4: a11(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00 (k=0, i=0)
    W[1][3] = 1  # C10 (k=0, i=1)
    
    # M5: (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # C00 (k=0, i=0)
    W[2][4] = 1  # C01 (k=1, i=0)
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # C11 (k=1, i=1)
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # C00 (k=0, i=0)
    W[2][6] = 1  # C01 (k=1, i=0)
    
    return U, V, W

def _block_decompose(n, m, p, block_n, block_m, block_p, block_rank):
    """
    Decompose n x m * m x p into blocks of size block_n x block_m * block_p.
    Each block uses a decomposition of rank block_rank.
    Total rank = (n/block_n) * (m/block_m) * (p/block_p) * block_rank
    """
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    
    num_blocks_i = n // block_n
    num_blocks_j = m // block_m
    num_blocks_k = p // block_p
    total_rank = num_blocks_i * num_blocks_j * num_blocks_k * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Get the base block decomposition
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_blocks_i):
        for bj in range(num_blocks_j):
            for bk in range(num_blocks_k):
                block_col_offset = (bi * num_blocks_j + bj) * num_blocks_k * block_rank + bk * block_rank
                
                for r in range(block_rank):
                    # U: rows are (i, j) where i in [bi*block_n, (bi+1)*block_n), j in [bj*block_m, (bj+1)*block_m)
                    for li in range(block_n):
                        for lj in range(block_m):
                            i = bi * block_n + li
                            j = bj * block_m + lj
                            u_val = base_U[li * block_m + lj][r]
                            if u_val != 0:
                                U[i * m + j][block_col_offset + r] = u_val
                    
                    # V: rows are (j, k) where j in [bj*block_m, (bj+1)*block_m), k in [bk*block_p, (bk+1)*block_p)
                    for lj in range(block_m):
                        for lk in range(block_p):
                            j = bj * block_m + lj
                            k = bk * block_p + lk
                            v_val = base_V[lj * block_p + lk][r]
                            if v_val != 0:
                                V[j * p + k][block_col_offset + r] = v_val
                    
                    # W: rows are (k, i) where k in [bk*block_p, (bk+1)*block_p), i in [bi*block_n, (bi+1)*block_n)
                    for lk in range(block_p):
                        for li in range(block_n):
                            k = bk * block_p + lk
                            i = bi * block_n + li
                            w_val = base_W[lk * block_n + li][r]
                            if w_val != 0:
                                W[k * n + i][block_col_offset + r] = w_val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try block decomposition with Strassen 2x2 blocks
    # For 4x4 x 4x8: blocks of 2x2 x 2x2
    # num_blocks_i = 4/2 = 2, num_blocks_j = 4/2 = 2, num_blocks_k = 8/2 = 4
    # total_rank = 2 * 2 * 4 * 7 = 112
    # Schoolbook rank = 4 * 4 * 8 = 128
    # So 112 < 128, this is an improvement!
    
    block_result = _block_decompose(n, m, p, 2, 2, 2, 7)
    if block_result is not None:
        block_rank = len(block_result["U"][0])
        if block_rank < best_rank:
            # Verify the decomposition
            if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
                best = block_result
                best_rank = block_rank
    
    return best
# EVOLVE-BLOCK-END
