"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 43 decomposition using 2x2x2 Strassen and schoolbook for the rest.
    # Split 3 = 2 + 1, 5 = 3 + 2.
    # Sub-blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:3] -> 2x2x3 (Rank 11)
    # 2. A[0:2, 2:3] * B[2:3, 0:3] -> 2x1x3 (Rank 6)
    # 3. A[2:3, 0:2] * B[0:2, 0:3] -> 1x2x3 (Rank 6)
    # 4. A[2:3, 2:3] * B[2:3, 0:3] -> 1x1x3 (Rank 3)
    # 5. A[0:2, 0:2] * B[0:2, 3:5] -> 2x2x2 (Rank 7)
    # 6. A[0:2, 2:3] * B[2:3, 3:5] -> 2x1x2 (Rank 4)
    # 7. A[2:3, 0:2] * B[0:2, 3:5] -> 1x2x2 (Rank 4)
    # 8. A[2:3, 2:3] * B[2:3, 3:5] -> 1x1x2 (Rank 2)
    
    # Total Rank = 11 + 6 + 6 + 3 + 7 + 4 + 4 + 2 = 43.
    
    n, m, p = DIMENSIONS
    total_rank = 43
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_block(U_sub, V_sub, W_sub, i_off, j_off, k_off, rank_sub):
        # U_sub: (n_sub * m_sub, rank_sub)
        # V_sub: (m_sub * p_sub, rank_sub)
        # W_sub: (p_sub * n_sub, rank_sub)
        # i_off, j_off: offset in A (n x m)
        # k_off: offset in B (m x p) for columns? No, B is m x p.
        # B indices: row j, col k.
        # A indices: row i, col j.
        # C indices: row i, col k.
        
        n_sub = len(U_sub) // m_sub if m_sub > 0 else 0
        # We need to know n_sub, m_sub, p_sub to map indices.
        # Let's pass dimensions explicitly.
        
    # Helper to generate schoolbook U, V, W for n x m x p
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Helper to generate Strassen 2x2x2 U, V, W
    def strassen_222():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        
        # U coefficients for A00, A01, A10, A11 (indices 0, 1, 2, 3)
        # V coefficients for B00, B01, B10, B11 (indices 0, 1, 2, 3)
        # W coefficients for C00, C01, C10, C11 (indices 0, 1, 2, 3)
        
        # Col 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1 # C00, C11? No.
        # C00 = M1... C11 = M1...
        # W[k*2+i]
        # C00 is i=0, k=0 -> W[0]
        # C11 is i=1, k=1 -> W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # Col 1: M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1 # C10, C11
        
        # Col 2: M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C01
        
        # Col 3: M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C00, C10
        
        # Col 4: M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C00, C01
        
        # Col 5: M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11
        
        # Col 6: M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[3][6] = 1 # C00, C11
        
        return U, V, W, 7

    def merge(U1, V1, W1, r1, U2, V2, W2, r2):
        r = r1 + r2
        U = [row[:] + [0]*r2 for row in U1] + [[0]*r1 + row[:] for row in U2]
        V = [row[:] + [0]*r2 for row in V1] + [[0]*r1 + row[:] for row in V2]
        W = [row[:] + [0]*r2 for row in W1] + [[0]*r1 + row[:] for row in W2]
        return U, V, W, r

    # 1. 2x2x3 (Rank 11)
    # Split B (2x3) into 2x2 and 2x1
    U_222, V_222, W_222, r_222 = strassen_222()
    U_221, V_221, W_221, r_221 = schoolbook(2, 2, 1)
    U_223, V_223, W_223, r_223 = merge(U_222, V_222, W_222, r_222, U_221, V_221, W_221, r_221)
    
    # 2. 2x1x3 (Rank 6)
    U_213, V_213, W_213, r_213 = schoolbook(2, 1, 3)
    
    # 3. 1x2x3 (Rank 6)
    U_123, V_123, W_123, r_123 = schoolbook(1, 2, 3)
    
    # 4. 1x1x3 (Rank 3)
    U_113, V_113, W_113, r_113 = schoolbook(1, 1, 3)
    
    # 5. 2x2x2 (Rank 7)
    U_222b, V_222b, W_222b, r_222b = strassen_222()
    
    # 6. 2x1x2 (Rank 4)
    U_212, V_212, W_212, r_212 = schoolbook(2, 1, 2)
    
    # 7. 1x2x2 (Rank 4)
    U_122, V_122, W_122, r_122 = schoolbook(1, 2, 2)
    
    # 8. 1x1x2 (Rank 2)
    U_112, V_112, W_112, r_112 = schoolbook(1, 1, 2)
    
    # Now assemble into global 3x3x5
    # Global U: 9x43, V: 15x43, W: 15x43
    
    def place_block(U_glob, V_glob, W_glob, U_sub, V_sub, W_sub, r_sub, i_off, j_off, k_off):
        # i_off: row offset in A (0..2)
        # j_off: col offset in A / row offset in B (0..2)
        # k_off: col offset in B (0..4)
        
        n_sub = len(U_sub) // (len(V_sub) // len(W_sub)) # This is tricky.
        # Better: pass n, m, p of the sub-block.
        
    # Let's just do it manually or with a clearer helper.
    
    def place(U_glob, V_glob, W_glob, U_sub, V_sub, W_sub, r_sub, n_sub, m_sub, p_sub, i_off, j_off, k_off):
        for i in range(n_sub):
            for j in range(m_sub):
                glob_u_idx = (i + i_off) * 3 + (j + j_off)
                for c in range(r_sub):
                    U_glob[glob_u_idx][c] = U_sub[i * m_sub + j][c]
        
        for j in range(m_sub):
            for k in range(p_sub):
                glob_v_idx = (j + j_off) * 5 + (k + k_off)
                for c in range(r_sub):
                    V_glob[glob_v_idx][c] = V_sub[j * p_sub + k][c]
        
        for k in range(p_sub):
            for i in range(n_sub):
                glob_w_idx = (k + k_off) * 3 + (i + i_off)
                for c in range(r_sub):
                    W_glob[glob_w_idx][c] = W_sub[k * n_sub + i][c]

    # Block 1: 2x2x3 at i=0, j=0, k=0
    place(U, V, W, U_223, V_223, W_223, r_223, 2, 2, 3, 0, 0, 0)
    # Block 2: 2x1x3 at i=0, j=2, k=0
    place(U, V, W, U_213, V_213, W_213, r_213, 2, 1, 3, 0, 2, 0)
    # Block 3: 1x2x3 at i=2, j=0, k=0
    place(U, V, W, U_123, V_123, W_123, r_123, 1, 2, 3, 2, 0, 0)
    # Block 4: 1x1x3 at i=2, j=2, k=0
    place(U, V, W, U_113, V_113, W_113, r_113, 1, 1, 3, 2, 2, 0)
    # Block 5: 2x2x2 at i=0, j=0, k=3
    place(U, V, W, U_222b, V_222b, W_222b, r_222b, 2, 2, 2, 0, 0, 3)
    # Block 6: 2x1x2 at i=0, j=2, k=3
    place(U, V, W, U_212, V_212, W_212, r_212, 2, 1, 2, 0, 2, 3)
    # Block 7: 1x2x2 at i=2, j=0, k=3
    place(U, V, W, U_122, V_122, W_122, r_122, 1, 2, 2, 2, 0, 3)
    # Block 8: 1x1x2 at i=2, j=2, k=3
    place(U, V, W, U_112, V_112, W_112, r_112, 1, 1, 2, 2, 2, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
