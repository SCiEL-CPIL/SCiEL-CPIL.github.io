"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2*2x2 matmul with rank 7 (Strassen).
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00 -> C10 += 1, C11 -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00*(B01-B11) -> C01 += 1, C11 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11*(B10-B00) -> C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11) -> C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _matmul_block(n, m, p, rank_override=None):
    """Return (U, V, W) for n x m * m x p matmul.
    If rank_override is given, use that rank (schoolbook).
    Otherwise, use best known decomposition.
    """
    if rank_override is not None:
        rank = rank_override
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # For specific small cases, use optimized decompositions
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # For other small cases, use schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _combine_blocks(blocks):
    """Combine a list of (U, V, W) blocks into a single decomposition.
    Each block has its own local indexing. We need to offset indices appropriately.
    This is handled by the caller providing pre-offset blocks.
    """
    total_rank = sum(len(W) for _, _, W in blocks)
    U_out = [[0]*total_rank for _ in range(max(len(U) for U, _, _ in blocks))]
    V_out = [[0]*total_rank for _ in range(max(len(V) for _, V, _ in blocks))]
    W_out = [[0]*total_rank for _ in range(max(len(_, _, W) for _, _, W in blocks))]
    
    offset = 0
    for U, V, W in blocks:
        for i in range(len(U)):
            for c in range(len(U[0])):
                U_out[i][offset + c] = U[i][c]
        for i in range(len(V)):
            for c in range(len(V[0])):
                V_out[i][offset + c] = V[i][c]
        for i in range(len(W)):
            for c in range(len(W[0])):
                W_out[i][offset + c] = W[i][c]
        offset += len(W[0])
    
    return U_out, V_out, W_out


def _make_block(n_local, m_local, p_local, i_off, j_off, k_off, rank_override=None):
    """Create a block for sub-problem A[i_off:i_off+n_local, j_off:j_off+m_local] * B[j_off:j_off+m_local, k_off:k_off+p_local].
    Returns (U, V, W) with global indexing.
    """
    U_local, V_local, W_local = _matmul_block(n_local, m_local, p_local, rank_override)
    
    n_total, m_total, p_total = 2, 3, 5
    U = [[0]*len(U_local[0]) for _ in range(n_total*m_total)]
    V = [[0]*len(V_local[0]) for _ in range(m_total*p_total)]
    W = [[0]*len(W_local[0]) for _ in range(p_total*n_total)]
    
    for i_l in range(n_local):
        for j_l in range(m_local):
            i_g = i_off + i_l
            j_g = j_off + j_l
            for c in range(len(U_local[0])):
                U[i_g*m_total + j_g][c] = U_local[i_l*m_local + j_l][c]
    
    for j_l in range(m_local):
        for k_l in range(p_local):
            j_g = j_off + j_l
            k_g = k_off + k_l
            for c in range(len(V_local[0])):
                V[j_g*p_total + k_g][c] = V_local[j_l*p_local + k_l][c]
    
    for k_l in range(p_local):
        for i_l in range(n_local):
            k_g = k_off + k_l
            i_g = i_off + i_l
            for c in range(len(W_local[0])):
                W[k_g*n_total + i_g][c] = W_local[k_l*n_local + i_l][c]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # Split the 3x5 problem into blocks:
    # A is 2x3, B is 3x5
    # Split A's columns into [0:2] and [2:3]
    # Split B's rows into [0:2] and [2:3]
    # Split B's columns into [0:2], [2:4], [4:5]
    
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> 2x2*2x2, rank 7
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> 2x2*2x2, rank 7
    # 3. A[0:2, 0:2] * B[0:2, 4:5] -> 2x2*2x1, rank 4
    # 4. A[0:2, 2:3] * B[2:3, 0:2] -> 2x1*1x2, rank 2
    # 5. A[0:2, 2:3] * B[2:3, 2:4] -> 2x1*1x2, rank 2
    # 6. A[0:2, 2:3] * B[2:3, 4:5] -> 2x1*1x1, rank 1
    
    # Wait, let me reconsider the splits.
    # A: 2 rows, 3 cols. Split cols: [0:2], [2:3]
    # B: 3 rows, 5 cols. Split rows: [0:2], [2:3]. Split cols: [0:2], [2:4], [4:5]
    
    # Block 1: i_off=0, j_off=0, k_off=0, n_l=2, m_l=2, p_l=2 -> Strassen, rank 7
    # Block 2: i_off=0, j_off=0, k_off=2, n_l=2, m_l=2, p_l=2 -> Strassen, rank 7
    # Block 3: i_off=0, j_off=0, k_off=4, n_l=2, m_l=2, p_l=1 -> schoolbook, rank 4
    # Block 4: i_off=0, j_off=2, k_off=0, n_l=2, m_l=1, p_l=2 -> schoolbook, rank 4
    # Block 5: i_off=0, j_off=2, k_off=2, n_l=2, m_l=1, p_l=2 -> schoolbook, rank 4
    # Block 6: i_off=0, j_off=2, k_off=4, n_l=2, m_l=1, p_l=1 -> schoolbook, rank 2
    
    # Total rank = 7 + 7 + 4 + 4 + 4 + 2 = 28
    
    blocks = []
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    U1, V1, W1 = _make_block(2, 2, 2, 0, 0, 0)
    blocks.append((U1, V1, W1))
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    U2, V2, W2 = _make_block(2, 2, 2, 0, 0, 2)
    blocks.append((U2, V2, W2))
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    U3, V3, W3 = _make_block(2, 2, 1, 0, 0, 4)
    blocks.append((U3, V3, W3))
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:2]
    U4, V4, W4 = _make_block(2, 1, 2, 0, 2, 0)
    blocks.append((U4, V4, W4))
    
    # Block 5: A[0:2, 2:3] * B[2:3, 2:4]
    U5, V5, W5 = _make_block(2, 1, 2, 0, 2, 2)
    blocks.append((U5, V5, W5))
    
    # Block 6: A[0:2, 2:3] * B[2:3, 4:5]
    U6, V6, W6 = _make_block(2, 1, 1, 0, 2, 4)
    blocks.append((U6, V6, W6))
    
    # Combine all blocks
    U, V, W = _combine_blocks(blocks)
    
    # Ensure correct dimensions
    assert len(U) == U_dim, f"U dim {len(U)} != {U_dim}"
    assert len(V) == V_dim, f"V dim {len(V)} != {V_dim}"
    assert len(W) == W_dim, f"W dim {len(W)} != {W_dim}"
    
    rank = len(W[0]) if W else 0
    assert len(U[0]) == rank, f"U cols {len(U[0])} != rank {rank}"
    assert len(V[0]) == rank, f"V cols {len(V[0])} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
