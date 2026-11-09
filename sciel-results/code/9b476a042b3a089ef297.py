"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank-44 Strassen-based decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 1x4 x 4x4 for row i=2 (16 terms)
    for j in range(4):
        for k in range(4):
            U[2 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    # Part 2: 2x4 x 4x4 for rows i=0,1 (28 terms)
    # Using Strassen for each 2x2x2x2 sub-block
    
    # I'll use the following rank-7 decomposition for 2x2x2x2:
    # This is a known valid decomposition.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]], Z = X@Y
    # 
    # The 7 rank-1 terms are:
    # t1: x00 * (y01+y11) -> z00 += 1, z01 += 1
    # t2: (x00+x01) * y10 -> z00 += -1, z10 += 1
    # t3: (x00+x10) * y01 -> z01 += 1, z11 += -1
    # t4: x11 * (y00+y01) -> z00 += 1, z11 += 1
    # t5: (x01+x11) * (y10+y11) -> z10 += 1, z11 += 1
    # t6: (x00-x10) * (y00+y01) -> (not used in output, but needed for rank-7)
    # t7: (x01-x11) * (y00+y10) -> (not used in output, but needed for rank-7)
    #
    # Wait, if t6 and t7 are not used in the output, then this is only a rank-5 decomposition,
    # which is impossible since the rank of 2x2x2x2 is 7.
    # 
    # I think the issue is that I'm not using the correct Strassen formulas.
    # Let me just use the schoolbook 8-term approach for now and accept rank 48.
    
    # Actually, let me just use the schoolbook for the 2x4x4 part (32 terms) + 16 for 1x4x4 = 48 total.
    # This is the same as the current program, so it won't improve the score.
    # 
    # Let me try a different approach: use the 2x2x2x2 rank-7 decomposition correctly.
    # I'll look up the correct formulas.
    
    # For now, let me just return the schoolbook decomposition.
    return schoolbook()
# EVOLVE-BLOCK-END
