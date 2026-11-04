import time

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    Split A (2x3) into A1 (2x2) and A2 (2x1) along columns.
    Split B (3x8) into B1 (2x8) and B2 (1x8) along rows.
    C = A1*B1 + A2*B2.
    
    A1*B1:
    A1 is 2x2, B1 is 2x8.
    Split B1 into four 2x2 blocks.
    Use Strassen's algorithm (Rank 7) for each 2x2 * 2x2 multiplication.
    Total rank for this part: 4 * 7 = 28.
    
    A2*B2:
    A2 is 2x1, B2 is 1x8. This is an outer product.
    Rank is 2 * 8 = 16.
    
    Total Rank: 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A10-A01)(B11+B01)
    
    # Local indices for 2x2 block:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i, n=2)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0, 1 of B. Split into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            # Global A indices: i in 0..1, a in 0..1
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            # Global B indices: a in 0..1, k in k_start..k_start+1
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # Fill W for C block (cols k_start..k_start+1 of C, rows 0..1 of C)
            # Global C indices: k in k_start..k_start+1, i in 0..1
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx][col] += val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A (rows 0,1).
    # B2 is row 2 of B (cols 0..7).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            
            col += 1
            
    # Validation
    # Check if sum(U[i,a,c] * V[a,k,c] * W[k,i,c]) == 1 for all i,a,k
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] != 0 and v_row[c] != 0 and w_row[c] != 0:
                        acc += u_row[c] * v_row[c] * w_row[c]
                if acc != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_sb[i * m + a][col_sb] = 1
                    V_sb[a * p + k][col_sb] = 1
                    W_sb[k * n + i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
