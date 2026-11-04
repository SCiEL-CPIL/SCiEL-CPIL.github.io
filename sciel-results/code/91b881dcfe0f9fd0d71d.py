"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
    So U is 2x2=4 rows, V is 2x2=4 rows, W is 2x2=4 rows
    """
    # Strassen 7 products for (A_{00} A_{01}; A_{10} A_{11}) * (B_{00} B_{01}; B_{10} B_{11})
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U rows: A_{i,j} for i in {0,1}, j in {0,1} -> index i*2+j
    # V rows: B_{j,k} for j in {0,1}, k in {0,1} -> index j*2+k
    # W rows: C_{k,i} for k in {0,1}, i in {0,1} -> index k*2+i
    
    U = [[Fraction(0) for _ in range(7)] for _ in range(4)]
    V = [[Fraction(0) for _ in range(7)] for _ in range(4)]
    W = [[Fraction(0) for _ in range(7)] for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[1][0] = 1  # C01
    W[2][0] = 1  # C10
    W[3][0] = 1  # C11
    
    # M2 = (A10+A11)(B00)
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = 1  # C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1 # B11
    W[0][2] = 1  # C00
    W[1][2] = 1  # C01
    
    # M4 = A11(B01-B00)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1 # B00
    W[2][3] = 1  # C10
    W[3][3] = 1  # C11
    
    # M5 = (A00+A01)(B11)
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[1][4] = 1  # C01
    W[3][4] = 1  # C11
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[2][5] = 1  # C10
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    W[1][6] = 1  # C01
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Return U, V, W for schoolbook decomposition of n x m x p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _embed(U, V, W, i_offset, j_offset, k_offset, n_sub, m_sub, p_sub, n_total, m_total, p_total):
    """Embed a sub-decomposition into the full n_total x m_total x p_total space.
    U_sub has shape (n_sub * m_sub, rank), V_sub has shape (m_sub * p_sub, rank), W_sub has shape (p_sub * n_sub, rank)
    Returns U_full (n_total * m_total, rank), V_full (m_total * p_total, rank), W_full (p_total * n_total, rank)
    """
    rank = len(U[0])
    U_full = [[Fraction(0) for _ in range(rank)] for _ in range(n_total * m_total)]
    V_full = [[Fraction(0) for _ in range(rank)] for _ in range(m_total * p_total)]
    W_full = [[Fraction(0) for _ in range(rank)] for _ in range(p_total * n_total)]
    
    for i_sub in range(n_sub):
        for j_sub in range(m_sub):
            row_sub = i_sub * m_sub + j_sub
            row_full = (i_offset + i_sub) * m_total + (j_offset + j_sub)
            for c in range(rank):
                U_full[row_full][c] = U[row_sub][c]
    
    for j_sub in range(m_sub):
        for k_sub in range(p_sub):
            row_sub = j_sub * p_sub + k_sub
            row_full = (j_offset + j_sub) * p_total + (k_offset + k_sub)
            for c in range(rank):
                V_full[row_full][c] = V[row_sub][c]
    
    for k_sub in range(p_sub):
        for i_sub in range(n_sub):
            row_sub = k_sub * n_sub + i_sub
            row_full = (k_offset + k_sub) * n_total + (i_offset + i_sub)
            for c in range(rank):
                W_full[row_full][c] = W[row_sub][c]
    
    return U_full, V_full, W_full


def _concat(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions along the rank dimension."""
    n1 = len(U1)
    m1 = len(V1)
    p1 = len(W1)
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(n1)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(m1)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(p1)]
    
    for i in range(n1):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
    
    for j in range(m1):
        for c in range(rank1):
            V[j][c] = V1[j][c]
        for c in range(rank2):
            V[j][rank1 + c] = V2[j][c]
    
    for k in range(p1):
        for c in range(rank1):
            W[k][c] = W1[k][c]
        for c in range(rank2):
            W[k][rank1 + c] = W2[k][c]
    
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    deadline = time.time() + 100  # 100 seconds budget, 20 seconds safety margin
    
    # Strategy: Split 4x5x5 into sub-blocks and use Strassen for 2x2 sub-blocks
    # Split i into {0,1} and {2,3}
    # Split j into {0,1}, {2,3}, {4}
    # Split k into {0,1}, {2,3}, {4}
    
    # Sub-problems:
    # 1. i={0,1}, j={0,1}, k={0,1}: 2x2x2 -> Strassen (rank 7)
    # 2. i={0,1}, j={0,1}, k={2,3}: 2x2x2 -> Strassen (rank 7)
    # 3. i={0,1}, j={0,1}, k={4}: 2x2x1 -> Schoolbook (rank 4)
    # 4. i={0,1}, j={2,3}, k={0,1}: 2x2x2 -> Strassen (rank 7)
    # 5. i={0,1}, j={2,3}, k={2,3}: 2x2x2 -> Strassen (rank 7)
    # 6. i={0,1}, j={2,3}, k={4}: 2x2x1 -> Schoolbook (rank 4)
    # 7. i={0,1}, j={4}, k={0,1}: 2x1x2 -> Schoolbook (rank 4)
    # 8. i={0,1}, j={4}, k={2,3}: 2x1x2 -> Schoolbook (rank 4)
    # 9. i={0,1}, j={4}, k={4}: 2x1x1 -> Schoolbook (rank 2)
    # 10. i={2,3}, j={0,1}, k={0,1}: 2x2x2 -> Strassen (rank 7)
    # 11. i={2,3}, j={0,1}, k={2,3}: 2x2x2 -> Strassen (rank 7)
    # 12. i={2,3}, j={0,1}, k={4}: 2x2x1 -> Schoolbook (rank 4)
    # 13. i={2,3}, j={2,3}, k={0,1}: 2x2x2 -> Strassen (rank 7)
    # 14. i={2,3}, j={2,3}, k={2,3}: 2x2x2 -> Strassen (rank 7)
    # 15. i={2,3}, j={2,3}, k={4}: 2x2x1 -> Schoolbook (rank 4)
    # 16. i={2,3}, j={4}, k={0,1}: 2x1x2 -> Schoolbook (rank 4)
    # 17. i={2,3}, j={4}, k={2,3}: 2x1x2 -> Schoolbook (rank 4)
    # 18. i={2,3}, j={4}, k={4}: 2x1x1 -> Schoolbook (rank 2)
    
    # Total rank: 7+7+4+7+7+4+4+4+2+7+7+4+7+7+4+4+4+2 = 96
    # Schoolbook rank: 4*5*5 = 100
    # Improvement: 100 - 96 = 4
    
    n_total, m_total, p_total = 4, 5, 5
    
    # Define sub-blocks
    # Each sub-block: (i_offset, i_size, j_offset, j_size, k_offset, k_size, use_strassen)
    sub_blocks = [
        (0, 2, 0, 2, 0, 2, True),   # 1
        (0, 2, 0, 2, 2, 2, True),   # 2
        (0, 2, 0, 2, 4, 1, False),  # 3
        (0, 2, 2, 2, 0, 2, True),   # 4
        (0, 2, 2, 2, 2, 2, True),   # 5
        (0, 2, 2, 2, 4, 1, False),  # 6
        (0, 2, 4, 1, 0, 2, False),  # 7
        (0, 2, 4, 1, 2, 2, False),  # 8
        (0, 2, 4, 1, 4, 1, False),  # 9
        (2, 2, 0, 2, 0, 2, True),   # 10
        (2, 2, 0, 2, 2, 2, True),   # 11
        (2, 2, 0, 2, 4, 1, False),  # 12
        (2, 2, 2, 2, 0, 2, True),   # 13
        (2, 2, 2, 2, 2, 2, True),   # 14
        (2, 2, 2, 2, 4, 1, False),  # 15
        (2, 2, 4, 1, 0, 2, False),  # 16
        (2, 2, 4, 1, 2, 2, False),  # 17
        (2, 2, 4, 1, 4, 1, False),  # 18
    ]
    
    # Accumulate U, V, W
    total_rank = sum(7 if sb[6] else sb[1] * sb[3] * sb[5] for sb in sub_blocks)
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(n_total * m_total)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(m_total * p_total)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(p_total * n_total)]
    
    col_offset = 0
    for (i_off, i_sz, j_off, j_sz, k_off, k_sz, use_strassen) in sub_blocks:
        if use_strassen:
            U_sub, V_sub, W_sub = _strassen_2x2x2()
            rank_sub = 7
        else:
            U_sub, V_sub, W_sub = _schoolbook_nm_p(i_sz, j_sz, k_sz)
            rank_sub = i_sz * j_sz * k_sz
        
        U_full, V_full, W_full = _embed(U_sub, V_sub, W_sub, i_off, j_off, k_off, i_sz, j_sz, k_sz, n_total, m_total, p_total)
        
        for i in range(n_total * m_total):
            for c in range(rank_sub):
                U[i][col_offset + c] += U_full[i][c]
        for j in range(m_total * p_total):
            for c in range(rank_sub):
                V[j][col_offset + c] += V_full[j][c]
        for k in range(p_total * n_total):
            for c in range(rank_sub):
                W[k][col_offset + c] += W_full[k][c]
        
        col_offset += rank_sub
        
        if time.time() > deadline:
            break
    
    # Convert to float for compatibility
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
