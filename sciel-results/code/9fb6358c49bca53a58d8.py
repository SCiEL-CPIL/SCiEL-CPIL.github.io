"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """Classic Strassen 2x2x2 decomposition, rank 7.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 * 2x2.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1   # A00, A11
    V[0][0] = 1; V[3][0] = 1   # B00, B11
    W[0][0] = 1; W[3][0] = 1   # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1   # A10, A11
    V[0][1] = 1                # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1                # A00
    V[2][2] = 1; V[3][2] = -1  # B10, B11
    W[1][2] = 1; W[3][2] = 1   # C01, C11
    
    # M4 = A11(B01-B10)
    U[3][3] = 1                # A11
    V[1][3] = 1; V[2][3] = -1  # B01, B10
    W[0][3] = 1; W[2][3] = 1   # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1   # A00, A01
    V[3][4] = 1                # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1   # B00, B01
    W[3][5] = 1                # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1   # B10, B11
    W[0][6] = 1                # C00
    
    return U, V, W, 7

def _merge_parts(parts, n, m, p):
    """
    Merge a list of sub-decompositions into a single decomposition.
    parts: list of (U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub, rank_sub)
    """
    total_rank = sum(part[8] for part in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub, rank_sub in parts:
        # U mapping
        for i_sub in range(n_sub):
            for j_sub in range(m_sub):
                sub_row = i_sub * m_sub + j_sub
                global_i = i_off + i_sub
                global_j = j_off + j_sub
                global_row = global_i * m + global_j
                for c in range(rank_sub):
                    if U_sub[sub_row][c] != 0:
                        U[global_row][col_offset + c] += U_sub[sub_row][c]
        
        # V mapping
        for j_sub in range(m_sub):
            for k_sub in range(p_sub):
                sub_row = j_sub * p_sub + k_sub
                global_j = j_off + j_sub
                global_k = k_off + k_sub
                global_row = global_j * p + global_k
                for c in range(rank_sub):
                    if V_sub[sub_row][c] != 0:
                        V[global_row][col_offset + c] += V_sub[sub_row][c]
        
        # W mapping
        for k_sub in range(p_sub):
            for i_sub in range(n_sub):
                sub_row = k_sub * n_sub + i_sub
                global_k = k_off + k_sub
                global_i = i_off + i_sub
                global_row = global_k * n + global_i
                for c in range(rank_sub):
                    if W_sub[sub_row][c] != 0:
                        W[global_row][col_offset + c] += W_sub[sub_row][c]
        
        col_offset += rank_sub
        
    return U, V, W, total_rank

def _get_decomp(n, m, p):
    """Recursively get the optimal decomposition for n x m * m x p."""
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)

    # Heuristic splits for specific cases
    # Try splitting m
    # We want to find a split m1 + m2 = m such that rank(n, m1, p) + rank(n, m2, p) < n*m*p
    # But we can just try a few specific splits that are known to work.
    
    # Strategy:
    # 1. If m is even, try m/2, m/2.
    # 2. If m is odd, try 1, m-1 or 2, m-2.
    # 3. If n is even, try n/2, n/2.
    # 4. If p is even, try p/2, p/2.
    
    # However, to ensure we hit the 108 rank path, we will check specific splits.
    
    candidates = []
    
    # Split m
    if m >= 2:
        # Try split m into 2 and m-2 if m>2, or 1 and m-1
        splits_m = []
        if m % 2 == 0:
            splits_m.append((m//2, m//2))
        else:
            splits_m.append((1, m-1))
            if m >= 3:
                splits_m.append((2, m-2))
        
        for m1, m2 in splits_m:
            if m1 < 1 or m2 < 1: continue
            U1, V1, W1, r1 = _get_decomp(n, m1, p)
            U2, V2, W2, r2 = _get_decomp(n, m2, p)
            # Offsets:
            # Part 1: j_off = 0
            # Part 2: j_off = m1
            # i_off = 0, k_off = 0 for both
            parts = [
                (U1, V1, W1, 0, 0, 0, n, m1, p, r1),
                (U2, V2, W2, 0, m1, 0, n, m2, p, r2)
            ]
            _, _, _, r_total = _merge_parts(parts, n, m, p)
            candidates.append((r_total, parts, 'm'))

    # Split n
    if n >= 2 and n % 2 == 0:
        n1 = n // 2
        n2 = n // 2
        U1, V1, W1, r1 = _get_decomp(n1, m, p)
        U2, V2, W2, r2 = _get_decomp(n2, m, p)
        parts = [
            (U1, V1, W1, 0, 0, 0, n1, m, p, r1),
            (U2, V2, W2, n1, 0, 0, n2, m, p, r2)
        ]
        _, _, _, r_total = _merge_parts(parts, n, m, p)
        candidates.append((r_total, parts, 'n'))

    # Split p
    if p >= 2:
        splits_p = []
        if p % 2 == 0:
            splits_p.append((p//2, p//2))
        else:
            splits_p.append((1, p-1))
            if p >= 3:
                splits_p.append((2, p-2))
        
        for p1, p2 in splits_p:
            if p1 < 1 or p2 < 1: continue
            U1, V1, W1, r1 = _get_decomp(n, m, p1)
            U2, V2, W2, r2 = _get_decomp(n, m, p2)
            parts = [
                (U1, V1, W1, 0, 0, 0, n, m, p1, r1),
                (U2, V2, W2, 0, 0, p1, n, m, p2, r2)
            ]
            _, _, _, r_total = _merge_parts(parts, n, m, p)
            candidates.append((r_total, parts, 'p'))

    if not candidates:
        return _schoolbook(n, m, p)

    # Pick the best candidate
    best_rank, best_parts, _ = min(candidates, key=lambda x: x[0])
    
    # If best_rank is not better than schoolbook, return schoolbook
    school_rank = n * m * p
    if best_rank >= school_rank:
        return _schoolbook(n, m, p)
    
    U, V, W, r = _merge_parts(best_parts, n, m, p)
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
