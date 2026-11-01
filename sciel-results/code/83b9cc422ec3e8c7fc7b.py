"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition."""

import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _build_strassen_2x2():
    """Return U, V, W matrices for 2x2 * 2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    W convention: W[k*n+i, c] where n=2
    """
    n = 2
    # Strassen's 7 multiplication terms for 2x2 matrices
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g)
    # M3 = (a-b)(e+f)
    # M4 = (a+d)(i+j)
    # M5 = (a+c)(i)
    # M6 = (b-d)(h+j)
    # M7 = (c-b)(e)
    
    # For A * B = C where A is 2x2, B is 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M6 + M7
    
    # U[i*m+j, c] for n=2, m=2: rows are a(0,0), b(0,1), c(1,0), d(1,1)
    U = [
        [1, 0, 1, 1, 1, 0, 0],  # a
        [0, 0, -1, 0, 0, 1, -1], # b
        [0, 1, 0, 0, 1, 0, -1],  # c
        [1, 1, 0, 1, 0, -1, 0],  # d
    ]
    
    # V[j*p+k, c] for m=2, p=2: rows are e(0,0), f(0,1), g(1,0), h(1,1)
    V = [
        [1, 0, 1, 0, 0, 0, 0],  # e
        [0, 0, 1, 0, 0, 0, 0],  # f
        [0, 1, 0, 0, 0, 0, 0],  # g
        [1, 0, 0, 0, 0, 1, 0],  # h
    ]
    
    # W[k*n+i, c] for p=2, n=2: rows are c00(0,0), c01(0,1), c10(1,0), c11(1,1)
    W = [
        [1, 0, 0, 1, -1, 0, -1],  # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 0, 0, 0, 1, 1],   # c11
    ]
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m * m x p."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, r

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple decompositions into one.
    Each decomposition is (U, V, W) for a sub-problem.
    The sub-problems partition the k-dimension of the output.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    U = [[0]*total_rank for _ in range(n_rows)]
    V = [[0]*total_rank for _ in range(m_rows)]
    W = [[0]*total_rank for _ in range(p_rows)]
    
    offset = 0
    for U_sub, V_sub, W_sub in decomps:
        r_sub = len(U_sub[0])
        for i in range(n_rows):
            for c in range(r_sub):
                U[i][offset + c] = U_sub[i][c]
        for i in range(m_rows):
            for c in range(r_sub):
                V[i][offset + c] = V_sub[i][c]
        for i in range(p_rows):
            for c in range(r_sub):
                W[i][offset + c] = W_sub[i][c]
        offset += r_sub
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=5
    # A is 2x2, B is 2x5, C = A*B is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strategy: Split B into B1 (2x2) and B2 (2x3)
    # C1 = A * B1 (2x2) - use Strassen, rank 7
    # C2 = A * B2 (2x3) - need to decompose this
    
    # For C2 = A * B2 where B2 is 2x3:
    # C2[i,k] = sum_j A[i,j] * B2[j,k]
    # This is a 2x2 * 2x3 multiplication
    # Schoolbook rank: 2*2*3 = 12
    # Can we do better?
    
    # Let's think about 2x2 * 2x3:
    # A = [[a, b], [c, d]], B2 = [[e, f, g], [h, i, j]]
    # C2 = [[c00, c01, c02], [c10, c11, c12]]
    # c00 = a*e + b*h
    # c01 = a*f + b*i
    # c02 = a*g + b*j
    # c10 = c*e + d*h
    # c11 = c*f + d*i
    # c12 = c*g + d*j
    
    # We can try to use Strassen-like ideas here too.
    # Note that B2 has 3 columns. Let's split it as B2 = [B2a | B2b] where B2a is 2x1 and B2b is 2x2.
    # Then C2 = [A*B2a | A*B2b]
    # A*B2b can use Strassen (rank 7)
    # A*B2a is 2x2 * 2x1, schoolbook rank 4
    
    # So total: 7 (Strassen for 2x2*2x2) + 4 (schoolbook for 2x2*2x1) = 11 for the 2x2*2x3 part
    # Plus 7 for the 2x2*2x2 part = 18 total
    
    # But wait, can we do better for 2x2*2x3?
    # Let's try to construct it directly.
    
    # Actually, let's try a different approach:
    # Split B (2x5) into B1 (2x2), B2 (2x2), B3 (2x1)
    # C = [A*B1 | A*B2 | A*B3]
    # Each A*B1 and A*B2 can use Strassen (rank 7 each)
    # A*B3 is 2x2*2x1 (rank 4)
    # Total: 7 + 7 + 4 = 18
    
    # Let's try to be smarter. Can we share some computation between the three blocks?
    
    # For now, let's implement the straightforward combination: Strassen for 2x2 blocks + schoolbook for 2x1
    
    # Build Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _build_strassen_2x2()
    
    # Build schoolbook for 2x2*2x1
    U_sb, V_sb, W_sb, _ = _schoolbook_block(2, 2, 1)
    
    # Now we need to combine these for the full 2x2*2x5
    # The decomposition should be:
    # U_full has 4 rows (for a, b, c, d)
    # V_full has 10 rows (for e, f, g, h, i, j, k, l, m, n where B = [[e,f,g,h,i],[j,k,l,m,n]])
    # W_full has 10 rows (for C[0,0], C[0,1], ..., C[1,4])
    
    # Let me re-think the structure.
    # A = [[a, b], [c, d]] (2x2)
    # B = [[b00, b01, b02, b03, b04],
    #      [b10, b11, b12, b13, b14]] (2x5)
    # C = [[c00, c01, c02, c03, c04],
    #      [c10, c11, c12, c13, c14]] (2x5)
    
    # c[i,k] = a*b[i,0] + c*b[i,1] ... wait no
    # c[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # So for each output column k, we're computing A * B[:,k]
    # B[:,k] is a 2x1 vector
    
    # Let's define:
    # For column k, let B_k = [B[0,k], B[1,k]]
    # C[:,k] = A * B_k
    
    # For 5 columns, we compute A * B_k for k=0,1,2,3,4
    
    # Using Strassen: A * X where X is 2x2 gives rank 7
    # For a single column (2x1), schoolbook gives rank 4
    
    # Strategy 1: Group columns into pairs and use Strassen
    # Columns 0,1 -> Strassen (rank 7)
    # Columns 2,3 -> Strassen (rank 7)  
    # Column 4 -> schoolbook (rank 4)
    # Total: 18
    
    # Let's build this:
    
    total_rank = 7 + 7 + 4
    n_rows = n * m  # 4
    m_rows = m * p  # 10
    p_rows = p * n  # 10
    
    U_full = [[0]*total_rank for _ in range(n_rows)]
    V_full = [[0]*total_rank for _ in range(m_rows)]
    W_full = [[0]*total_rank for _ in range(p_rows)]
    
    # Block 1: columns 0,1 (Strassen)
    offset = 0
    # U: rows 0-3 (a,b,c,d)
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    
    # V: rows for B[0,0], B[0,1], B[1,0], B[1,1]
    # V_full row j*p+k where j is row of B (0 or 1), k is column within the 2x2 block
    # B[0,0] -> row 0*5+0 = 0
    # B[0,1] -> row 0*5+1 = 1
    # B[1,0] -> row 1*5+0 = 5
    # B[1,1] -> row 1*5+1 = 6
    
    # In the Strassen V matrix, the rows correspond to e, f, g, h
    # e = B[0,0] -> V_full row 0
    # f = B[0,1] -> V_full row 1
    # g = B[1,0] -> V_full row 5
    # h = B[1,1] -> V_full row 6
    
    v_row_map = [0, 1, 5, 6]  # maps Strassen V row index to V_full row index
    for i in range(4):
        for c in range(7):
            V_full[v_row_map[i]][offset + c] = V_s[i][c]
    
    # W: rows for C[0,0], C[0,1], C[1,0], C[1,1]
    # W_full row k*n+i where k is output column, i is output row
    # C[0,0] -> row 0*2+0 = 0
    # C[0,1] -> row 0*2+1 = 1
    # C[1,0] -> row 1*2+0 = 2
    # C[1,1] -> row 1*2+1 = 3
    
    w_row_map = [0, 1, 2, 3]
    for i in range(4):
        for c in range(7):
            W_full[w_row_map[i]][offset + c] = W_s[i][c]
    
    offset += 7
    
    # Block 2: columns 2,3 (Strassen)
    for i in range(4):
        for c in range(7):
            U_full[i][offset + c] = U_s[i][c]
    
    # B[0,2] -> row 0*5+2 = 2
    # B[0,3] -> row 0*5+3 = 3
    # B[1,2] -> row 1*5+2 = 7
    # B[1,3] -> row 1*5+3 = 8
    
    v_row_map2 = [2, 3, 7, 8]
    for i in range(4):
        for c in range(7):
            V_full[v_row_map2[i]][offset + c] = V_s[i][c]
    
    # C[0,2] -> row 2*2+0 = 4
    # C[0,3] -> row 2*2+1 = 5
    # C[1,2] -> row 3*2+0 = 6
    # C[1,3] -> row 3*2+1 = 7
    
    w_row_map2 = [4, 5, 6, 7]
    for i in range(4):
        for c in range(7):
            W_full[w_row_map2[i]][offset + c] = W_s[i][c]
    
    offset += 7
    
    # Block 3: column 4 (schoolbook 2x2*2x1)
    # U_sb rows: a(0), b(1), c(2), d(3)
    # V_sb rows: B[0,4](0), B[1,4](1)
    # W_sb rows: C[0,4](0), C[1,4](1)
    
    for i in range(4):
        for c in range(4):
            U_full[i][offset + c] = U_sb[i][c]
    
    # B[0,4] -> row 0*5+4 = 4
    # B[1,4] -> row 1*5+4 = 9
    for i in range(2):
        for c in range(4):
            V_full[i*5+4][offset + c] = V_sb[i][c]
    
    # C[0,4] -> row 4*2+0 = 8
    # C[1,4] -> row 4*2+1 = 9
    for i in range(2):
        for c in range(4):
            W_full[i*2+4][offset + c] = W_sb[i][c]
    
    # Verify
    if _verify(U_full, V_full, W_full, n, m, p, total_rank):
        # Convert to clean format
        def frac_to_clean(f):
            if isinstance(f, Fraction):
                if f.denominator == 1:
                    return int(f)
                else:
                    return float(f)
            return f
        
        U_out = [[frac_to_clean(x) for x in row] for row in U_full]
        V_out = [[frac_to_clean(x) for x in row] for row in V_full]
        W_out = [[frac_to_clean(x) for x in row] for row in W_full]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook
    U_sb_full, V_sb_full, W_sb_full, r = _schoolbook_block(n, m, p)
    return {"U": U_sb_full, "V": V_sb_full, "W": W_sb_full}

# EVOLVE-BLOCK-END
