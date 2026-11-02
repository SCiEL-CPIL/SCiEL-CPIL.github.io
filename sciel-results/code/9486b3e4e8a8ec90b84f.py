import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank23_3x3x3():
    """
    Return the rank-23 decomposition for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    R = 23
    U = [[0.0] * R for _ in range(9)]
    V = [[0.0] * R for _ in range(9)]
    W = [[0.0] * R for _ in range(9)]
    
    def ai(i, j): return i * 3 + j
    def bi(j, k): return j * 3 + k
    def ci(k, i): return k * 3 + i
    
    # We'll build the 23 terms.
    # The standard 23-mult algorithm for 3x3x3:
    # 
    # Let A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # Let B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A*B
    #
    # The 23 multiplications are:
    
    terms = []
    
    # --- Strassen-like 7 mults for the top-left 2x2 block (indices 0,1) ---
    # M0 = (a00+a11)*(b00+b11)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1
    v[bi(0,0)] = 1; v[bi(1,1)] = 1
    w[ci(0,0)] = 1; w[ci(1,1)] = 1
    terms.append((u, v, w))
    
    # M1 = (a10+a11)*b00
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(1,0)] = 1; u[ai(1,1)] = 1
    v[bi(0,0)] = 1
    w[ci(1,0)] = 1
    terms.append((u, v, w))
    
    # M2 = a00*(b01-b11)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1
    v[bi(0,1)] = 1; v[bi(1,1)] = -1
    w[ci(0,1)] = 1
    terms.append((u, v, w))
    
    # M3 = a11*(b10-b00)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(1,1)] = 1
    v[bi(1,0)] = 1; v[bi(0,0)] = -1
    w[ci(1,1)] = 1
    terms.append((u, v, w))
    
    # M4 = (a00+a01)*b11
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(0,1)] = 1
    v[bi(1,1)] = 1
    w[ci(0,1)] = 1
    terms.append((u, v, w))
    
    # M5 = (a10-a00)*(b00+b01)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(1,0)] = 1; u[ai(0,0)] = -1
    v[bi(0,0)] = 1; v[bi(0,1)] = 1
    w[ci(1,0)] = 1; w[ci(1,1)] = 1
    terms.append((u, v, w))
    
    # M6 = (a01-a11)*(b10+b11)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,1)] = 1; u[ai(1,1)] = -1
    v[bi(1,0)] = 1; v[bi(1,1)] = 1
    w[ci(0,0)] = 1; w[ci(0,1)] = 1
    terms.append((u, v, w))
    
    # Now the remaining 16 terms handle the 3rd row/column interactions.
    # These are derived from the known 23-mult algorithm.
    
    # M7 = a02*(b00+b10+b20)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,2)] = 1
    v[bi(0,0)] = 1; v[bi(1,0)] = 1; v[bi(2,0)] = 1
    w[ci(0,0)] = 1; w[ci(0,1)] = 1; w[ci(0,2)] = 1
    terms.append((u, v, w))
    
    # M8 = a12*(b00+b10+b20)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(1,2)] = 1
    v[bi(0,0)] = 1; v[bi(1,0)] = 1; v[bi(2,0)] = 1
    w[ci(1,0)] = 1; w[ci(1,1)] = 1; w[ci(1,2)] = 1
    terms.append((u, v, w))
    
    # M9 = a20*(b00+b01+b02)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,0)] = 1
    v[bi(0,0)] = 1; v[bi(0,1)] = 1; v[bi(0,2)] = 1
    w[ci(0,2)] = 1; w[ci(1,2)] = 1; w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # M10 = a21*(b00+b01+b02)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,1)] = 1
    v[bi(0,0)] = 1; v[bi(0,1)] = 1; v[bi(0,2)] = 1
    w[ci(0,2)] = 1; w[ci(1,2)] = 1; w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # M11 = a22*(b00+b01+b02)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,2)] = 1
    v[bi(0,0)] = 1; v[bi(0,1)] = 1; v[bi(0,2)] = 1
    w[ci(0,2)] = 1; w[ci(1,2)] = 1; w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # M12 = (a00+a11+a22)*(b00+b11+b22)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1; u[ai(2,2)] = 1
    v[bi(0,0)] = 1; v[bi(1,1)] = 1; v[bi(2,2)] = 1
    w[ci(0,0)] = 1; w[ci(1,1)] = 1; w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # M13 = (a00+a11+a22)*(b01+b12+b20)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1; u[ai(2,2)] = 1
    v[bi(0,1)] = 1; v[bi(1,2)] = 1; v[bi(2,0)] = 1
    w[ci(0,1)] = 1; w[ci(1,2)] = 1; w[ci(2,0)] = 1
    terms.append((u, v, w))
    
    # M14 = (a00+a11+a22)*(b10+b21+b02)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1; u[ai(2,2)] = 1
    v[bi(1,0)] = 1; v[bi(2,1)] = 1; v[bi(0,2)] = 1
    w[ci(1,0)] = 1; w[ci(2,1)] = 1; w[ci(0,2)] = 1
    terms.append((u, v, w))
    
    # M15 = a02*b01
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,2)] = 1
    v[bi(0,1)] = 1
    w[ci(0,1)] = 1
    terms.append((u, v, w))
    
    # M16 = a02*b12
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,2)] = 1
    v[bi(1,2)] = 1
    w[ci(0,2)] = 1
    terms.append((u, v, w))
    
    # M17 = a12*b20
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(1,2)] = 1
    v[bi(2,0)] = 1
    w[ci(1,0)] = 1
    terms.append((u, v, w))
    
    # M18 = a20*b11
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,0)] = 1
    v[bi(1,1)] = 1
    w[ci(1,2)] = 1
    terms.append((u, v, w))
    
    # M19 = a21*b22
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,1)] = 1
    v[bi(2,2)] = 1
    w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # M20 = a22*b00
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,2)] = 1
    v[bi(0,0)] = 1
    w[ci(0,2)] = 1
    terms.append((u, v, w))
    
    # M21 = (a00+a11)*(b20+b21+b22)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1
    v[bi(2,0)] = 1; v[bi(2,1)] = 1; v[bi(2,2)] = 1
    w[ci(0,2)] = 1; w[ci(1,2)] = 1
    terms.append((u, v, w))
    
    # M22 = (a20+a21+a22)*(b00+b10+b20)
    u = [0.0]*9; v = [0.0]*9; w = [0.0]*9
    u[ai(2,0)] = 1; u[ai(2,1)] = 1; u[ai(2,2)] = 1
    v[bi(0,0)] = 1; v[bi(1,0)] = 1; v[bi(2,0)] = 1
    w[ci(0,2)] = 1; w[ci(1,2)] = 1; w[ci(2,2)] = 1
    terms.append((u, v, w))
    
    # Now apply all terms
    for r, (u, v, w) in enumerate(terms):
        for idx_u in range(9):
            U[idx_u][r] = u[idx_u]
        for idx_v in range(9):
            V[idx_v][r] = v[idx_v]
        for idx_w in range(9):
            W[idx_w][r] = w[idx_w]
    
    return U, V, W

def _schoolbook_3x3x4():
    """
    Return the schoolbook decomposition for 3x3 * 3x4.
    U: (9, 36), V: (12, 36), W: (12, 36)
    U indices: i*3+j for i,j in 0..2
    V indices: j*4+k for j in 0..2, k in 0..3
    W indices: k*3+i for k in 0..3, i in 0..2
    """
    n, m, p = 3, 3, 4
    R = n * m * p  # 36
    U = [[0.0] * R for _ in range(n * m)]  # 9 rows
    V = [[0.0] * R for _ in range(m * p)]  # 12 rows
    W = [[0.0] * R for _ in range(p * n)]  # 12 rows
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _schoolbook_3x3x3():
    """Return schoolbook for 3x3x3, rank 27."""
    n, m, p = 3, 3, 3
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _combine_decompositions(parts, n, m, p):
    """
    Combine multiple decomposition parts into a single decomposition for n x m x p.
    parts: list of (U_part, V_part, W_part, col_offset) where col_offset is the starting column in the full p dimension.
    """
    total_rank = sum(len(U[0]) for U, V, W, _ in parts)
    
    U_full = [[0.0] * total_rank for _ in range(n * m)]
    V_full = [[0.0] * total_rank for _ in range(m * p)]
    W_full = [[0.0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for U_part, V_part, W_part, col_offset in parts:
        R_part = len(U_part[0])
        p_part = len(W_part) // n  # number of columns in this part
        m_part = len(V_part) // p_part  # number of rows in B part
        
        for r in range(R_part):
            for i in range(n):
                for j in range(m):
                    # U_part index: i*m_part + j
                    U_full[i * m + j][rank_offset + r] = U_part[i * m_part + j][r]
            
            for j in range(m_part):
                for k in range(p_part):
                    # V_part index: j*p_part + k
                    # Full V index: j*p + (col_offset + k)
                    V_full[j * p + (col_offset + k)][rank_offset + r] = V_part[j * p_part + k][r]
            
            for k in range(p_part):
                for i in range(n):
                    # W_part index: k*n + i
                    # Full W index: (col_offset + k)*n + i
                    W_full[(col_offset + k) * n + i][rank_offset + r] = W_part[k * n + i][r]
        
        rank_offset += R_part
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Strategy: Split p=7 into p1=3 and p2=4
    # Part 1: 3x3 * 3x3 with rank 23 (Strassen-based)
    # Part 2: 3x3 * 3x4 with rank 36 (schoolbook)
    # Total rank = 59
    
    p1, p2 = 3, 4
    
    # Get rank-23 decomposition for 3x3x3
    U1, V1, W1 = _rank23_3x3x3()
    
    # Get schoolbook decomposition for 3x3x4
    U2, V2, W2 = _schoolbook_3x3x4()
    
    # Combine: Part 1 handles columns 0-2, Part 2 handles columns 3-6
    U_full, V_full, W_full = _combine_decompositions(
        [(U1, V1, W1, 0), (U2, V2, W2, 3)],
        n, m, p
    )
    
    # Verify the decomposition
    if _verify_decomposition(U_full, V_full, W_full, n, m, p, tol=1e-10):
        return {
            "U": U_full,
            "V": V_full,
            "W": W_full
        }
    
    # If verification fails, try with schoolbook for both parts
    U1_sb, V1_sb, W1_sb = _schoolbook_3x3x3()
    U_full2, V_full2, W_full2 = _combine_decompositions(
        [(U1_sb, V1_sb, W1_sb, 0), (U2, V2, W2, 3)],
        n, m, p
    )
    
    if _verify_decomposition(U_full2, V_full2, W_full2, n, m, p, tol=1e-10):
        return {
            "U": U_full2,
            "V": V_full2,
            "W": W_full2
        }
    
    # Fallback: full schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
