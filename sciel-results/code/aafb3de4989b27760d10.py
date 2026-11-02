"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: Use Winograd-style or improved decomposition for blocks
    # For 2x2 * 2x3: can we do rank 7?
    # Let's try a direct construction for 2x2 * 2x3
    # A is 2x2: [a00, a01; a10, a11]
    # B is 2x3: [b00, b01, b02; b10, b11, b12]
    # C is 2x3: [c00, c01, c02; c10, c11, c12]
    
    # Winograd for 2x2 * 2x2 has rank 7 (Strassen)
    # For 2x2 * 2x3, let's try rank 8 or 9
    
    # Actually, let's use a known result: the rank of M_{2,2,3} is 7
    # This is because we can use the same Strassen factors and just add one more column
    # Wait, that would give rank 7 + 4 = 11 which is worse than 2*7=14
    
    # Let me think differently. For 2x2 * 2xp where p is odd:
    # We can pair up columns and use Strassen for pairs, with one leftover
    # For p=13: 6 pairs + 1 single = 6*7 + 4 = 46 (current)
    
    # Alternative: Use a different decomposition
    # For 2x2 * 2x3, the rank is actually 7 (same as 2x2*2x2)
    # This is because the 3rd column can be "absorbed" into the same multiplications
    
    # Let me construct a rank-7 decomposition for 2x2 * 2x3
    # Using the Strassen formula but modified:
    # P1 = (a00+a11)(b00+b11+b12)  -- modified to include b12
    # Actually this gets complicated. Let me try a different approach.
    
    # Use block structure with blocks of size 4:
    # 13 = 4*3 + 1
    # If rank of M_{2,2,4} is 10, then 3*10 + 4 = 34
    # If rank of M_{2,2,4} is 12, then 3*12 + 4 = 40
    
    # Let me try to construct M_{2,2,4} with rank 10
    # A is 2x2, B is 2x4, C is 2x4
    # Using two Strassen decompositions for two 2x2 blocks would give rank 14
    # Can we do better?
    
    # Actually, let's use a completely different approach:
    # Use the fact that for 2x2 * 2xp, we can use p independent Strassen decompositions
    # but share the A-side factors
    
    # New idea: Use rank-4 schoolbook but with clever sharing
    # For each column j, we need 4 multiplications: a00*b0j, a01*b1j, a10*b0j, a11*b1j
    # Total: 13*4 = 52 (schoolbook)
    # Strassen gives 7 per 2 columns: 6*7 + 4 = 46
    
    # Let me try to find a better decomposition using the algebra structure
    # Represent B as a polynomial: B(x) = B0 + B1*x where B0, B1 are 2-vectors
    # Then A*B(x) = A*B0 + A*B1*x
    # This doesn't directly help
    
    # Given the complexity, let me stick with the current approach but try blocks of 3
    # For 2x2 * 2x3, I'll try rank 7
    # If it works: 4*7 + 4 = 32 (4 blocks of 3 + 1 single)
    # 13 = 3*4 + 1
    
    # Constructing rank-7 for 2x2 * 2x3:
    # This is equivalent to 2x2 * 2x2 with an extra column
    # The rank cannot be less than the rank of 2x2*2x2 which is 7
    # Can we achieve exactly 7?
    
    # Let's use the following construction:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b10-b11)
    # m4 = a11*(b01-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    
    # This gives us C for the first 2 columns. For the 3rd column (b02, b12):
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    
    # We need 4 more multiplications for the 3rd column, giving rank 11
    # Unless we can share...
    
    # Alternative: Use a different set of 7 multiplications that covers all 3 columns
    # This seems unlikely without more analysis
    
    # Given the time constraint, let me implement the current best (rank 46)
    # but with a cleaner structure and verification
    
    # Strassen decomposition for 2x2 * 2x2
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ])
    strassen_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ])
    strassen_W = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [1, -1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1, 0, 0],
        [1, 1, 0, -1, 0, -1, 0],
    ])
    
    # Verify Strassen decomposition numerically
    S_target = np.zeros((4, 4, 4), dtype=float)
    S_target[0, 0, 0] = 1  # a00 * b00 -> C00
    S_target[0, 1, 2] = 1  # a00 * b10 -> C10
    S_target[1, 2, 0] = 1  # a01 * b01 -> C01
    S_target[1, 3, 2] = 1  # a01 * b11 -> C11
    S_target[2, 0, 1] = 1  # a10 * b00 -> C01
    S_target[2, 1, 3] = 1  # a10 * b01 -> C11
    S_target[3, 2, 1] = 1  # a11 * b10 -> C01
    S_target[3, 3, 3] = 1  # a11 * b11 -> C11
    
    S_recon = np.zeros((4, 4, 4), dtype=float)
    for t in range(7):
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    S_recon[a, b, c] += strassen_U[a][t] * strassen_V[b][t] * strassen_W[c][t]
    
    if not np.allclose(S_recon, S_target, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build full decomposition with 6 Strassen blocks + 1 schoolbook
    total_rank = 6 * 7 + 4  # = 46
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for block in range(6):
        j0 = 2 * block
        for t in range(7):
            U_full[:, col] = strassen_U[:, t]
            for k in range(2):
                for jl in range(2):
                    j_global = j0 + jl
                    v_local = k * 2 + jl
                    v_global = k * p + j_global
                    V_full[v_global, col] = strassen_V[v_local, t]
            for jl in range(2):
                for i in range(2):
                    j_global = j0 + jl
                    w_local = jl * 2 + i
                    w_global = j_global * n + i
                    W_full[w_global, col] = strassen_W[w_local, t]
            col += 1
    
    j = 12
    for i in range(n):
        for k in range(m):
            U_full[i * m + k, col] = 1
            V_full[k * p + j, col] = 1
            W_full[j * n + i, col] = 1
            col += 1
    
    # Verify full decomposition
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1
    
    T_recon = np.zeros((n * m, m * p, p * n), dtype=float)
    for t in range(total_rank):
        for a in range(n * m):
            ua = U_full[a, t]
            if ua == 0:
                continue
            for b in range(m * p):
                ub = ua * V_full[b, t]
                if ub == 0:
                    continue
                for c in range(p * n):
                    T_recon[a, b, c] += ub * W_full[c, t]
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
