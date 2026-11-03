"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-42 decomposition for 2x4 by 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    A (2x4) is split into 2 blocks of 2x2 along columns.
    B (4x6) is split into 3 blocks of 2x2 along columns.
    Each block pair uses 7 Strassen multiplications.
    Total rank = 2 * 3 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # X (2x2) * Y (2x2) = Z (2x2)
    # P1 = (X00+X11)*(Y00+Y11)
    # P2 = (X10+X11)*Y00
    # P3 = X00*(Y01-Y11)
    # P4 = X11*(Y10-Y00)
    # P5 = (X00+X01)*Y11
    # P6 = (X10-X00)*(Y00+Y01)
    # P7 = (X01-X11)*(Y10+Y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # U coefficients: X (first matrix, 2x2), indexed by (i, j)
    U_strassen = [
        {(0,0): 1, (1,1): 1},       # P1: X00 + X11
        {(1,0): 1, (1,1): 1},       # P2: X10 + X11
        {(0,0): 1},                  # P3: X00
        {(1,1): 1},                  # P4: X11
        {(0,0): 1, (0,1): 1},       # P5: X00 + X01
        {(1,0): 1, (0,0): -1},      # P6: X10 - X00
        {(0,1): 1, (1,1): -1}       # P7: X01 - X11
    ]
    
    # V coefficients: Y (second matrix, 2x2), indexed by (j, k)
    V_strassen = [
        {(0,0): 1, (1,1): 1},       # P1: Y00 + Y11
        {(0,0): 1},                  # P2: Y00
        {(0,1): 1, (1,1): -1},      # P3: Y01 - Y11
        {(1,0): 1, (0,0): -1},      # P4: Y10 - Y00
        {(1,1): 1},                  # P5: Y11
        {(0,0): 1, (0,1): 1},       # P6: Y00 + Y01
        {(1,0): 1, (1,1): 1}        # P7: Y10 + Y11
    ]
    
    # W coefficients: Z (output, 2x2), indexed by (k, i) per trace convention W[k*n+i]
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    W_strassen = [
        {(0,0): 1, (1,1): 1},       # P1: Z00(+1), Z11(+1)
        {(1,0): 1, (1,1): -1},      # P2: Z10(+1), Z11(-1)
        {(0,1): 1, (1,1): 1},       # P3: Z01(+1), Z11(+1)
        {(0,0): 1, (1,0): 1},       # P4: Z00(+1), Z10(+1)
        {(0,0): -1, (0,1): 1},      # P5: Z00(-1), Z01(+1)
        {(1,1): 1},                 # P6: Z11(+1)
        {(0,0): 1}                  # P7: Z00(+1)
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)
    
    # j_groups: split the 4 columns of A (rows of B) into 2 groups of 2
    j_groups = [(0, 1), (2, 3)]
    # k_groups: split the 6 columns of B into 3 groups of 2
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    col = 0
    for jg in j_groups:
        j_base = jg[0]  # 0 or 2
        for kg in k_groups:
            k_base = kg[0]  # 0, 2, or 4
            for s in range(7):
                # Update U: A indices (i, j) where j = j_base + j_local
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = j_base + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] = coeff
                
                # Update V: B indices (j, k) where j = j_base + j_local, k = k_base + k_local
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = j_base + j_loc
                    k = k_base + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] = coeff
                
                # Update W: output indices (k, i) where k = k_base + k_local, i = i_loc
                # Trace convention: W[k*n + i]
                for (k_loc, i_loc), coeff in W_strassen[s].items():
                    k = k_base + k_loc
                    i = i_loc
                    w_idx = k * n + i
                    W[w_idx][col] = coeff
                
                col += 1
    
    # Verify with random tests
    import random
    for trial in range(20):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for c in range(rank):
            u_val = sum(U[i * m + j][c] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][c] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][c] * u_val * v_val
        
        if C_comp != C_exp:
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
