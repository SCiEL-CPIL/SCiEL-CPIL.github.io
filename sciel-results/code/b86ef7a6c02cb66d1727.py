"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition for 2x2 x 2x2 -> 2x2.
    U: (2*2)x7, V: (2*2)x7, W: (2*2)x7
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # Result: [[P1+P4+P5-P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # U[i*2+j] corresponds to A[i][j]
    # V[j*2+k] corresponds to B[j][k]
    # W[k*2+i] corresponds to C[k][i]... wait, W[k*n+i] with n=2
    
    # Let me use the standard Strassen formulation.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*2+j, c] = coefficient of A[i][j] in product c
    # V[j*2+k, c] = coefficient of B[j][k] in product c
    # W[k*2+i, c] = coefficient of C[k][i]... 
    # Wait, the convention is W[k*n+i, r] where n=2, so W[k*2+i, c]
    # C[i][j] = sum_k A[i][k]*B[k][j]
    # So W should map to C indices. Let's say W[i*2+j, c] for C[i][j]?
    # But the spec says W[k*n+i, r]. With n=2, that's W[k*2+i, r].
    # This means W is indexed by (k, i) where k is the column of C (from B's row) and i is the row of C (from A's row).
    # So W[k*2+i, c] corresponds to C[i][k]? No...
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # In the schoolbook: W[k*n+i][column] = 1, and the check is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # So W[k*n+i, c] is the coefficient for the (i, k) entry of the output C.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # So for 2x2x2: W[k*2+i, c] is coeff for C[i][k].
    
    # m1 = (A00+A11)*(B00+B11)
    # U: A00->1, A11->1; V: B00->1, B11->1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += m1, C11 += m1
    W[0*2+0][0] = 1  # C[0][0]
    W[1*2+1][0] = 1  # C[1][1]
    
    # m2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += m2, C11 -= m2
    W[0*2+1][1] = 1  # C[1][0]
    W[1*2+1][1] = -1  # C[1][1]
    
    # m3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 += m3, C11 += m3
    W[1*2+0][2] = 1  # C[0][1]
    W[1*2+1][2] = 1  # C[1][1]
    
    # m4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 += m4, C10 += m4
    W[0*2+0][3] = 1  # C[0][0]
    W[0*2+1][3] = 1  # C[1][0]
    
    # m5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 -= m5, C01 += m5
    W[0*2+0][4] = -1  # C[0][0]
    W[1*2+0][4] = 1  # C[0][1]
    
    # m6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += m6
    W[1*2+1][5] = 1  # C[1][1]
    
    # m7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 -= m7
    W[0*2+0][6] = -1  # C[0][0]
    
    return U, V, W

def _trivial_2x2x1():
    """Return trivial rank-4 decomposition for 2x2 x 2x1 -> 2x1.
    U: (2*2)x4, V: (2*1)x4, W: (1*2)x4
    """
    n, m, p = 2, 2, 1
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

def _combine_block_decompositions(blocks):
    """Combine independent block decompositions into one.
    Each block is (U_block, V_block, W_block) with shapes matching subproblems.
    We need to map subproblem indices to global indices.
    
    For 2x2x9: We can split the p=9 dimension into blocks.
    Block 1: 2x2x2 (rows 0-1 of A, cols 0-1 of B) -> uses Strassen rank 7
    Block 2: 2x2x2 (rows 0-1 of A, cols 2-3 of B) -> uses Strassen rank 7
    Block 3: 2x2x2 (rows 0-1 of A, cols 4-5 of B) -> uses Strassen rank 7
    Block 4: 2x2x2 (rows 0-1 of A, cols 6-7 of B) -> uses Strassen rank 7
    Block 5: 2x2x1 (rows 0-1 of A, col 8 of B) -> uses trivial rank 4
    Total rank = 7*4 + 4 = 32
    """
    n, m, p = 2, 2, 9
    total_rank = sum(b[0].shape[1] if hasattr(b[0], 'shape') else len(b[0][0]) for b in blocks)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for block in blocks:
        U_b, V_b, W_b = block
        rb = len(U_b[0])
        
        # U_b has shape (n_sub * m_sub) x rb
        # For our case, n_sub=2, m_sub=2, so U_b is 4xrb
        # Global U index: i*m + j where i is global row, j is global col of A
        # Since we're splitting along p (columns of B), the A indices are the same.
        
        # V_b has shape (m_sub * p_sub) x rb
        # Global V index: j*p + k where j is row of B (= col of A), k is col of B
        # We need to know which k range this block covers.
        
        # W_b has shape (p_sub * n_sub) x rb
        # Global W index: k*n + i where k is col of B, i is row of C (= row of A)
        
        # For simplicity, assume blocks are ordered by k range.
        # Block r covers k in [k_start, k_end)
        
        U_b_rows = len(U_b)
        V_b_rows = len(V_b)
        W_b_rows = len(W_b)
        
        # Determine p_sub from V_b: V_b has m_sub * p_sub rows
        # m_sub = 2 (same as global m since we don't split m)
        p_sub = V_b_rows // 2
        
        # Determine k_start from W_b: W_b has p_sub * n_sub rows
        # The first row of W_b corresponds to k=k_start, i=0
        # We'll figure out k_start from the block order
        
        # Actually, let's pass k_start explicitly or compute from position
        # Let me restructure: blocks should include their k offset
        
        for r in range(rb):
            for i in range(U_b_rows):
                if U_b[i][r] != 0:
                    U[i][col_offset + r] = U_b[i][r]
            for j in range(V_b_rows):
                if V_b[j][r] != 0:
                    # j is local index in V_b, which is j_local * p_sub + k_local
                    # Global V index = j_global * p + k_global
                    # j_global = j_local (since m is not split)
                    # k_global = k_start + k_local
                    j_local = j // p_sub
                    k_local = j % p_sub
                    k_global = k_start + k_local
                    V[j_local * p + k_global][col_offset + r] = V_b[j][r]
            for w in range(W_b_rows):
                if W_b[w][r] != 0:
                    # w is local index in W_b, which is k_local * n_sub + i_local
                    # Global W index = k_global * n + i_global
                    k_local = w // 2  # n_sub = 2
                    i_local = w % 2
                    k_global = k_start + k_local
                    W[k_global * n + i_local][col_offset + r] = W_b[w][r]
        
        col_offset += rb
        k_start += p_sub
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Build block decomposition: 4 Strassen 2x2x2 blocks + 1 trivial 2x2x1 block
    strassen_blocks = []
    for _ in range(4):
        U_s, V_s, W_s = _strassen_2x2x2()
        strassen_blocks.append((U_s, V_s, W_s))
    
    U_t, V_t, W_t = _trivial_2x2x1()
    trivial_block = (U_t, V_t, W_t)
    
    blocks = strassen_blocks + [trivial_block]
    
    U, V, W = _combine_block_decompositions(blocks)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
