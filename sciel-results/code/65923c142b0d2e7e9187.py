import time

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 blocks for the 2x2 * 2x8 sub-problem
    and schoolbook for the 2x1 * 1x8 sub-problem.
    """
    n, m, p = 2, 3, 8
    
    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Rank 44 Construction
    # A (2x3) = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B (3x8) = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x8. Split B1 into four 2x2 blocks.
    # Each 2x2 * 2x2 uses Strassen (rank 7). Total 4*7 = 28.
    # A2*B2: 2x1 * 1x8. Schoolbook. Total 2*8 = 16.
    # Total rank: 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    # Strassen 2x2x2 decomposition coefficients
    # Local indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    S_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    S_W = [
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 consists of rows 0 and 1 of B.
    # B1 is split into 4 blocks of 2 columns each.
    for block_idx in range(4):
        k_start = block_idx * 2
        k_end = k_start + 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_end)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # Fill W for C block (cols k_start..k_end of C, rows 0..1 of C)
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
        return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))

BET: {mechanism: "The program correctly constructs a rank-44 decomposition using Strassen for the 2x2*2x8 part and schoolbook for 2x1*1x8, which is the same algorithm as the current seed, ensuring a valid and correct result that ties the current score.", q: 0.95, delta: [0, 0]}
