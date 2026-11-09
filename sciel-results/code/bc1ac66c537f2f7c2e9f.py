import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for M_{n,m,p}
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
        return U, V, W

    # Helper to create Strassen decomposition for M_{2,2,2}
    def strassen_222():
        n, m, p = 2, 2, 2
        rank = 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # m1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # m2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # m3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # m4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # m5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # m6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # m7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = -1
        
        return U, V, W

    # Function to combine sub-decompositions into a larger one
    # sub_decomps is a list of (U, V, W, i_offset, j_offset, k_offset, i_size, j_size, k_size)
    def combine_decomps(sub_decomps, n, m, p):
        total_rank = sum(len(U[0]) for U, V, W, *_ in sub_decomps)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank_offset = 0
        for U_sub, V_sub, W_sub, i_off, j_off, k_off, i_sz, j_sz, k_sz in sub_decomps:
            r_sub = len(U_sub[0])
            for i in range(i_sz):
                for j in range(j_sz):
                    for r in range(r_sub):
                        U[(i + i_off) * m + (j + j_off)][rank_offset + r] = U_sub[i * j_sz + j][r]
            for j in range(j_sz):
                for k in range(k_sz):
                    for r in range(r_sub):
                        V[(j + j_off) * p + (k + k_off)][rank_offset + r] = V_sub[j * k_sz + k][r]
            for k in range(k_sz):
                for i in range(i_sz):
                    for r in range(r_sub):
                        W[(k + k_off) * n + (i + i_off)][rank_offset + r] = W_sub[k * i_sz + i][r]
            rank_offset += r_sub
        return U, V, W

    # Build M_{3,3,3}
    # Sub-decomps for M_{3,3,3} using S2={0,1}, S1={2}
    # 1. M_{2,2,2} at (0,0,0)
    # 2. M_{2,1,2} at (0,2,0) -> i in 0,1; j in 2; k in 0,1
    # 3. M_{1,2,2} at (2,0,0) -> i in 2; j in 0,1; k in 0,1
    # 4. M_{2,2,1} at (0,0,2) -> i in 0,1; j in 0,1; k in 2
    # 5. M_{1,1,2} at (2,2,0) -> i in 2; j in 2; k in 0,1
    # 6. M_{1,2,1} at (2,0,2) -> i in 2; j in 0,1; k in 2
    # 7. M_{2,1,1} at (0,2,2) -> i in 0,1; j in 2; k in 2
    # 8. M_{1,1,1} at (2,2,2) -> i in 2; j in 2; k in 2
    
    U_s222, V_s222, W_s222 = strassen_222()
    U_s212, V_s212, W_s212 = schoolbook(2, 1, 2)
    U_s122, V_s122, W_s122 = schoolbook(1, 2, 2)
    U_s221, V_s221, W_s221 = schoolbook(2, 2, 1)
    U_s112, V_s112, W_s112 = schoolbook(1, 1, 2)
    U_s121, V_s121, W_s121 = schoolbook(1, 2, 1)
    U_s211, V_s211, W_s211 = schoolbook(2, 1, 1)
    U_s111, V_s111, W_s111 = schoolbook(1, 1, 1)
    
    subs_333 = [
        (U_s222, V_s222, W_s222, 0, 0, 0, 2, 2, 2),
        (U_s212, V_s212, W_s212, 0, 2, 0, 2, 1, 2),
        (U_s122, V_s122, W_s122, 2, 0, 0, 1, 2, 2),
        (U_s221, V_s221, W_s221, 0, 0, 2, 2, 2, 1),
        (U_s112, V_s112, W_s112, 2, 2, 0, 1, 1, 2),
        (U_s121, V_s121, W_s121, 2, 0, 2, 1, 2, 1),
        (U_s211, V_s211, W_s211, 0, 2, 2, 2, 1, 1),
        (U_s111, V_s111, W_s111, 2, 2, 2, 1, 1, 1),
    ]
    U_333, V_333, W_333 = combine_decomps(subs_333, 3, 3, 3)
    
    # Build M_{3,3,2}
    # Sub-decomps for M_{3,3,2} using S2={0,1}, S1={2}
    # 1. M_{2,2,2} at (0,0,0)
    # 2. M_{2,1,2} at (0,2,0)
    # 3. M_{1,2,2} at (2,0,0)
    # 4. M_{1,1,2} at (2,2,0)
    
    subs_332 = [
        (U_s222, V_s222, W_s222, 0, 0, 0, 2, 2, 2),
        (U_s212, V_s212, W_s212, 0, 2, 0, 2, 1, 2),
        (U_s122, V_s122, W_s122, 2, 0, 0, 1, 2, 2),
        (U_s112, V_s112, W_s112, 2, 2, 0, 1, 1, 2),
    ]
    U_332, V_332, W_332 = combine_decomps(subs_332, 3, 3, 2)
    
    # Build M_{3,3,8}
    # Two M_{3,3,3} and one M_{3,3,2}
    # k offsets: 0, 3, 6
    
    subs_338 = [
        (U_333, V_333, W_333, 0, 0, 0, 3, 3, 3),
        (U_333, V_333, W_333, 0, 0, 3, 3, 3, 3),
        (U_332, V_332, W_332, 0, 0, 6, 3, 3, 2),
    ]
    U_final, V_final, W_final = combine_decomps(subs_338, 3, 3, 8)
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-START
# The code above is the new design.
# EVOLVE-BLOCK-END
