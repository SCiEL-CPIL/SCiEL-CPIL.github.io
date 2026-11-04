import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation
    # R(3,4,6) <= 2 * R(3,2,6)
    # R(3,2,6) <= 2 * R(3,2,3)
    # R(3,2,3) = 7 + 4 + 4 + 2 = 17
    # Total Rank = 2 * 2 * 17 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a sub-decomposition
    # local_n, local_m, local_p are dimensions of the sub-problem
    # i_offset, j_offset, k_offset are offsets in the global indices
    # U_sub, V_sub, W_sub are the local decompositions (lists of lists)
    # start_rank is the offset in the rank dimension
    
    def add_decomp(local_n, local_m, local_p, i_off, j_off, k_off, U_sub, V_sub, W_sub, start_rank):
        for i in range(local_n):
            for j in range(local_m):
                gi = i + i_off
                gj = j + j_off
                for r in range(len(U_sub[0])):
                    U[gi * local_m + j][start_rank + r] += U_sub[i * local_m + j][r]
                    # Note: U index is i*local_m + j in local, but in global it's (i+i_off)*m + (j+j_off)?
                    # No, U shape is (n*m, rank). Index is i*m + j.
                    # So U[gi*m + gj][start_rank + r] += ...
                    
        # Correcting U index logic
        # U is global. Index is i_global * m + j_global.
        # i_global = i + i_off. j_global = j + j_off.
        
        for i in range(local_n):
            for j in range(local_m):
                gi = i + i_off
                gj = j + j_off
                idx_u = gi * m + gj
                for r in range(len(U_sub[0])):
                    U[idx_u][start_rank + r] += U_sub[i * local_m + j][r]
                    
        for j in range(local_m):
            for k in range(local_p):
                gj = j + j_off
                gk = k + k_off
                idx_v = gj * p + gk
                for r in range(len(V_sub[0])):
                    V[idx_v][start_rank + r] += V_sub[j * local_p + k][r]
                    
        for i in range(local_n):
            for k in range(local_p):
                gi = i + i_off
                gk = k + k_off
                idx_w = gk * n + gi # Convention W[k*n+i]
                for r in range(len(W_sub[0])):
                    W[idx_w][start_rank + r] += W_sub[k * local_n + i][r]
                    
    # 1. Define Strassen 2x2x2
    def strassen_222():
        rn = 2; rm = 2; rp = 2
        r_rank = 7
        u = [[0]*r_rank for _ in range(rn*rm)]
        v = [[0]*r_rank for _ in range(rm*rp)]
        w = [[0]*r_rank for _ in range(rp*rn)]
        
        # P1
        u[0*2+0][0] = 1; u[1*2+1][0] = 1
        v[0*2+0][0] = 1; v[1*2+1][0] = 1
        w[0*2+0][0] = 1 # C00
        w[1*2+1][0] = 1 # C11
        
        # P2
        u[1*2+0][1] = 1; u[1*2+1][1] = 1
        v[0*2+0][1] = 1
        w[1*2+0][1] = 1 # C10
        
        # P3
        u[0*2+0][2] = 1
        v[0*2+1][2] = 1; v[1*2+1][2] = -1
        w[0*2+1][2] = 1 # C01
        w[1*2+1][2] = 1 # C11
        
        # P4
        u[1*2+1][3] = 1
        v[1*2+0][3] = 1; v[0*2+0][3] = -1
        w[0*2+0][3] = 1 # C00
        w[1*2+0][3] = 1 # C10
        
        # P5
        u[0*2+0][4] = 1; u[0*2+1][4] = 1
        v[1*2+1][4] = 1
        w[0*2+0][4] = -1 # C00
        w[0*2+1][4] = 1 # C01
        
        # P6
        u[1*2+0][5] = 1; u[0*2+0][5] = -1
        v[0*2+0][5] = 1; v[0*2+1][5] = 1
        w[1*2+1][5] = 1 # C11
        
        # P7
        u[0*2+1][6] = 1; u[1*2+1][6] = -1
        v[1*2+0][6] = 1; v[1*2+1][6] = 1
        w[0*2+0][6] = 1 # C00
        w[1*2+1][6] = 1 # C11
        
        return u, v, w

    # 2. Define Schoolbook 2x2x1
    def school_221():
        rn = 2; rm = 2; rp = 1
        r_rank = 4
        u = [[0]*r_rank for _ in range(rn*rm)]
        v = [[0]*r_rank for _ in range(rm*rp)]
        w = [[0]*r_rank for _ in range(rp*rn)]
        
        # C0 = A00 B0 + A01 B1
        u[0*2+0][0] = 1; v[0*1+0][0] = 1; w[0*2+0][0] = 1
        u[0*2+1][1] = 1; v[1*1+0][1] = 1; w[0*2+0][1] = 1
        
        # C1 = A10 B0 + A11 B1
        u[1*2+0][2] = 1; v[0*1+0][2] = 1; w[0*2+1][2] = 1
        u[1*2+1][3] = 1; v[1*1+0][3] = 1; w[0*2+1][3] = 1
        
        return u, v, w

    # 3. Define Schoolbook 1x2x2
    def school_122():
        rn = 1; rm = 2; rp = 2
        r_rank = 4
        u = [[0]*r_rank for _ in range(rn*rm)]
        v = [[0]*r_rank for _ in range(rm*rp)]
        w = [[0]*r_rank for _ in range(rp*rn)]
        
        # C0 = A0 B00 + A1 B10
        u[0*2+0][0] = 1; v[0*2+0][0] = 1; w[0*1+0][0] = 1
        u[0*2+1][1] = 1; v[1*2+0][1] = 1; w[0*1+0][1] = 1
        
        # C1 = A0 B01 + A1 B11
        u[0*2+0][2] = 1; v[0*2+1][2] = 1; w[1*1+0][2] = 1
        u[0*2+1][3] = 1; v[1*2+1][3] = 1; w[1*1+0][3] = 1
        
        return u, v, w

    # 4. Define Schoolbook 1x2x1
    def school_121():
        rn = 1; rm = 2; rp = 1
        r_rank = 2
        u = [[0]*r_rank for _ in range(rn*rm)]
        v = [[0]*r_rank for _ in range(rm*rp)]
        w = [[0]*r_rank for _ in range(rp*rn)]
        
        # C0 = A0 B0 + A1 B1
        u[0*2+0][0] = 1; v[0*1+0][0] = 1; w[0*1+0][0] = 1
        u[0*2+1][1] = 1; v[1*1+0][1] = 1; w[0*1+0][1] = 1
        
        return u, v, w

    # Assemble 3x2x3
    def get_323():
        u = [[0]*17 for _ in range(3*2)]
        v = [[0]*17 for _ in range(2*3)]
        w = [[0]*17 for _ in range(3*3)]
        
        # Sub 1: 2x2x2 at i:0-1, j:0-1, k:0-1
        u1, v1, w1 = strassen_222()
        add_decomp(2, 2, 2, 0, 0, 0, u1, v1, w1, 0)
        
        # Sub 2: 2x2x1 at i:0-1, j:0-1, k:2
        u2, v2, w2 = school_221()
        add_decomp(2, 2, 1, 0, 0, 2, u2, v2, w2, 7)
        
        # Sub 3: 1x2x2 at i:2, j:0-1, k:0-1
        u3, v3, w3 = school_122()
        add_decomp(1, 2, 2, 2, 0, 0, u3, v3, w3, 11)
        
        # Sub 4: 1x2x1 at i:2, j:0-1, k:2
        u4, v4, w4 = school_121()
        add_decomp(1, 2, 1, 2, 0, 2, u4, v4, w4, 15)
        
        return u, v, w

    # Assemble 3x2x6
    def get_326():
        u = [[0]*34 for _ in range(3*2)]
        v = [[0]*34 for _ in range(2*6)]
        w = [[0]*34 for _ in range(6*3)]
        
        # Sub 1: 3x2x3 at k:0-2
        u1, v1, w1 = get_323()
        add_decomp(3, 2, 3, 0, 0, 0, u1, v1, w1, 0)
        
        # Sub 2: 3x2x3 at k:3-5
        u2, v2, w2 = get_323()
        add_decomp(3, 2, 3, 0, 0, 3, u2, v2, w2, 17)
        
        return u, v, w

    # Assemble 3x4x6
    # Sub 1: 3x2x6 at j:0-1
    u1, v1, w1 = get_326()
    add_decomp(3, 2, 6, 0, 0, 0, u1, v1, w1, 0)
    
    # Sub 2: 3x2x6 at j:2-3
    u2, v2, w2 = get_326()
    add_decomp(3, 2, 6, 0, 2, 0, u2, v2, w2, 34)
    
    return {"U": U, "V": V, "W": W}
