import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the known rank-39 decomposition
    # 5 blocks of 7 (for column pairs 0-1, 2-3, 4-5, 6-7, 8-9) + 4 for column 10
    rank_fallback = 5 * 7 + 4
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U_fallback[0][r] = 1; U_fallback[3][r] = 1
        V_fallback[v_e][r] = 1; V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = 1; W_fallback[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U_fallback[2][r] = 1; U_fallback[3][r] = 1
        V_fallback[v_e][r] = 1
        W_fallback[w_c10][r] = 1; W_fallback[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U_fallback[0][r] = 1
        V_fallback[v_f][r] = 1; V_fallback[v_h][r] = -1
        W_fallback[w_c01][r] = 1; W_fallback[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U_fallback[3][r] = 1
        V_fallback[v_g][r] = 1; V_fallback[v_e][r] = -1
        W_fallback[w_c00][r] = 1; W_fallback[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U_fallback[0][r] = 1; U_fallback[1][r] = 1
        V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = -1; W_fallback[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U_fallback[2][r] = 1; U_fallback[0][r] = -1
        V_fallback[v_e][r] = 1; V_fallback[v_f][r] = 1
        W_fallback[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U_fallback[1][r] = 1; U_fallback[3][r] = -1
        V_fallback[v_g][r] = 1; V_fallback[v_h][r] = 1
        W_fallback[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U_fallback[0][r] = 1
    V_fallback[v_e2][r] = 1
    W_fallback[w_c02][r] = 1
    r += 1
    
    U_fallback[1][r] = 1
    V_fallback[v_g2][r] = 1
    W_fallback[w_c02][r] = 1
    r += 1
    
    U_fallback[2][r] = 1
    V_fallback[v_e2][r] = 1
    W_fallback[w_c12][r] = 1
    r += 1
    
    U_fallback[3][r] = 1
    V_fallback[v_g2][r] = 1
    W_fallback[w_c12][r] = 1
    r += 1
    
    # Verify the fallback is correct
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r_ in range(rank):
                        s += U[a][r_] * V[b][r_] * W[c][r_]
                    if s != 1:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # Key insight: Can we merge the last single column (rank 4) with one of the pairs (rank 7)
    # to get rank 10 for a 2x2x3 block? If so, total rank = 4*7 + 10 = 38
    
    # For 2x2x3, the standard rank is 8 (schoolbook) or potentially lower
    # Let's try to find a rank-10 decomposition for 2x2x3
    
    # Actually, let me think about this differently.
    # The current construction uses rank 7 for each 2x2x2 block.
    # For 2x2x2, the minimal rank is 7 (Strassen).
    # For 2x2x3, what's the minimal rank?
    # Schoolbook is 8. Can we do better?
    
    # Let's try a different approach: use a 2x2x3 construction with rank 9
    # Then total = 4*7 + 9 = 37
    
    # Or even better, can we do 2x2x3 with rank 8?
    # Then total = 4*7 + 8 = 36
    
    # Let me try to construct a 2x2x3 rank-8 decomposition
    # A is 2x2: [[a, b], [c, d]]
    # B is 2x3: [[e, f, g], [h, i, j]]
    # C is 2x3: [[c00, c01, c02], [c10, c11, c12]]
    
    # Schoolbook:
    # c00 = ae + bh, c01 = af + bi, c02 = ag + bj
    # c10 = ce + dh, c11 = cf + di, c12 = cg + dj
    
    # That's 8 multiplications. Can we share any?
    # For 2x2x2, Strassen uses 7. For 2x2x3, maybe we can use 7 as well?
    # Or 8 is minimal?
    
    # Let's try to use the Strassen-like approach for 2x2x3
    # Use the 2x2x2 Strassen for the first two columns, then add the third column
    
    # Actually, let me just try to verify if rank 38 is achievable by a different construction
    
    # For now, return the known rank-39 decomposition
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
