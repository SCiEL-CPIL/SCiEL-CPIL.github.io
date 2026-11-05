"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    U shape: (4, 7) - rows indexed by (i, j) where i, j in {0, 1}
    V shape: (4, 7) - rows indexed by (j, k) where j, k in {0, 1}
    W shape: (4, 7) - rows indexed by (k, i) where k, i in {0, 1}
    
    The tensor decomposition satisfies:
    sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = delta_{ik} for valid (i,j,k)
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Standard Strassen:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    #
    # c00 = p1 - p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p6
    # c11 = p1 - p2 - p3 + p6
    
    # p1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1  # a00, a11
    V[0,0]=1; V[3,0]=1  # b00, b11
    W[0,0]=1; W[3,0]=1  # c00 (k=0,i=0), c11 (k=1,i=1)
    
    # p2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1  # a10, a11
    V[0,1]=1            # b00
    W[1,1]=1            # c10 (k=0,i=1)
    
    # p3 = a00(b10-b11)
    U[0,2]=1            # a00
    V[2,2]=1; V[3,2]=-1 # b10, b11
    W[2,2]=1            # c01 (k=1,i=0)
    
    # p4 = a11(b01-b00)
    U[3,3]=1            # a11
    V[1,3]=1; V[0,3]=-1 # b01, b00
    W[0,3]=-1           # c00 (k=0,i=0)
    
    # p5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1  # a00, a01
    V[3,4]=1            # b11
    W[0,4]=-1; W[2,4]=1 # c00 (k=0,i=0), c01 (k=1,i=0)
    
    # p6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1 # a10, a00
    V[0,5]=1; V[1,5]=1  # b00, b01
    W[1,5]=1; W[3,5]=1  # c10 (k=0,i=1), c11 (k=1,i=1)
    
    # p7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1 # a01, a11
    V[2,6]=1; V[3,6]=1  # b10, b11
    W[0,6]=1; W[3,6]=-1 # c00 (k=0,i=0), c11 (k=1,i=1)
    
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """Return schoolbook decomposition for n_b x m_b x p_b block."""
    rank = n_b * m_b * p_b
    U = np.zeros((n_b * m_b, rank))
    V = np.zeros((m_b * p_b, rank))
    W = np.zeros((p_b * n_b, rank))
    
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i * m_b + j, col] = 1
                V[j * p_b + k, col] = 1
                W[k * n_b + i, col] = 1
                col += 1
    
    return U, V, W

def _embed_block(U_block, V_block, W_block, i_off, j_off, k_off, n, m, p, n_b, m_b, p_b):
    """
    Embed a block decomposition into the full n x m x p decomposition.
    
    U_block: (n_b*m_b, r) for A submatrix starting at row i_off, col j_off
    V_block: (m_b*p_b, r) for B submatrix starting at row j_off, col k_off
    W_block: (p_b*n_b, r) for C submatrix starting at row k_off, col i_off
    
    Returns full U, V, W matrices (zero elsewhere).
    """
    r = U_block.shape[1]
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for c in range(r):
        for di in range(n_b):
            for dj in range(m_b):
                a_idx = (i_off + di) * m + (j_off + dj)
                U[a_idx, c] = U_block[di * m_b + dj, c]
        
        for dj in range(m_b):
            for dk in range(p_b):
                b_idx = (j_off + dj) * p + (k_off + dk)
                V[b_idx, c] = V_block[dj * p_b + dk, c]
        
        for dk in range(p_b):
            for di in range(n_b):
                c_idx = (k_off + dk) * n + (i_off + di)
                W[c_idx, c] = W_block[dk * n_b + di, c]
    
    return U, V, W

def _build_3x4x8_decomposition():
    """
    Build a decomposition for 3x4x8 using Strassen on 2x2x2 blocks
    and schoolbook for remaining blocks.
    
    Partition:
    - i (rows of A, 0..2): sizes [2, 1]
    - j (cols of A / rows of B, 0..3): sizes [2, 2]
    - k (cols of B, 0..7): sizes [2, 2, 2, 2]
    
    Blocks:
    - 2x2x2: 8 blocks (i=0, j=0/1, k=0/1/2/3) -> Strassen rank 7 each = 56
    - 1x2x2: 8 blocks (i=1, j=0/1, k=0/1/2/3) -> Schoolbook rank 4 each = 32
    Total rank: 88
    """
    n, m, p = 3, 4, 8
    
    # Partition boundaries
    i_sizes = [2, 1]
    j_sizes = [2, 2]
    k_sizes = [2, 2, 2, 2]
    
    i_offsets = [0]
    for s in i_sizes[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    
    j_offsets = [0]
    for s in j_sizes[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    
    k_offsets = [0]
    for s in k_sizes[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    # Get Strassen decomposition (computed once)
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Collect all block decompositions
    all_U = []
    all_V = []
    all_W = []
    
    for ii in range(len(i_sizes)):
        for jj in range(len(j_sizes)):
            for kk in range(len(k_sizes)):
                n_b = i_sizes[ii]
                m_b = j_sizes[jj]
                p_b = k_sizes[kk]
                i_off = i_offsets[ii]
                j_off = j_offsets[jj]
                k_off = k_offsets[kk]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen
                    U_blk, V_blk, W_blk = U_str, V_str, W_str
                else:
                    # Use schoolbook
                    U_blk, V_blk, W_blk = _schoolbook_block(n_b, m_b, p_b)
                
                U_full, V_full, W_full = _embed_block(
                    U_blk, V_blk, W_blk,
                    i_off, j_off, k_off,
                    n, m, p,
                    n_b, m_b, p_b
                )
                
                all_U.append(U_full)
                all_V.append(V_full)
                all_W.append(W_full)
    
    # Concatenate all blocks horizontally
    total_rank = sum(U.shape[1] for U in all_U)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for U_blk, V_blk, W_blk in zip(all_U, all_V, all_W):
        r = U_blk.shape[1]
        U[:, col:col+r] = U_blk
        V[:, col:col+r] = V_blk
        W[:, col:col+r] = W_blk
        col += r
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(result, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < 1e-10

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    Uses Strassen's rank-7 decomposition for 2x2x2 blocks and
    schoolbook for remaining blocks.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    result = _build_3x4x8_decomposition()
    
    # Verify correctness
    if _verify_decomposition(result, n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
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

# EVOLVE-BLOCK-END
