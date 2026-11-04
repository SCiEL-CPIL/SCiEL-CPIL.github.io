"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-66 tensor decomposition for 3x4 * 4x6 matrix multiplication
    using Strassen (rank 7) for 2x2x2 blocks and direct (rank 4) for 1x2x2 blocks.
    """
    n, m, p = 3, 4, 6
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        U_terms.append(u_entries)
        V_terms.append(v_entries)
        W_terms.append(w_entries)
    
    def strassen_2x2x2(a_rows, a_cols, b_cols):
        a0, a1 = a_rows
        b0, b1 = a_cols
        c0, c1 = b_cols
        
        # m1 = (A[a0,b0]+A[a1,b1]) * (B[b0,c0]+B[b1,c1])
        # c00 += m1, c11 += m1
        add_term(
            [(a0*m + b0, 1), (a1*m + b1, 1)],
            [(b0*p + c0, 1), (b1*p + c1, 1)],
            [(c0*n + a0, 1), (c1*n + a1, 1)]
        )
        
        # m2 = (A[a1,b0]+A[a1,b1]) * B[b0,c0]
        # c10 += m2, c11 -= m2
        add_term(
            [(a1*m + b0, 1), (a1*m + b1, 1)],
            [(b0*p + c0, 1)],
            [(c0*n + a1, 1), (c1*n + a1, -1)]
        )
        
        # m3 = A[a0,b0] * (B[b1,c0]-B[b1,c1])
        # c01 += m3, c11 += m3
        add_term(
            [(a0*m + b0, 1)],
            [(b1*p + c0, 1), (b1*p + c1, -1)],
            [(c1*n + a0, 1), (c1*n + a1, 1)]
        )
        
        # m4 = A[a1,b1] * (B[b0,c1]-B[b0,c0])
        # c00 -= m4, c01 += m4
        add_term(
            [(a1*m + b1, 1)],
            [(b0*p + c1, 1), (b0*p + c0, -1)],
            [(c0*n + a0, -1), (c1*n + a0, 1)]
        )
        
        # m5 = (A[a0,b0]+A[a0,b1]) * B[b1,c1]
        # c00 += m5, c10 += m5
        add_term(
            [(a0*m + b0, 1), (a0*m + b1, 1)],
            [(b1*p + c1, 1)],
            [(c0*n + a0, 1), (c0*n + a1, 1)]
        )
        
        # m6 = (A[a1,b0]-A[a0,b0]) * (B[b0,c0]+B[b0,c1])
        # c00 += m6, c10 += m6
        add_term(
            [(a1*m + b0, 1), (a0*m + b0, -1)],
            [(b0*p + c0, 1), (b0*p + c1, 1)],
            [(c0*n + a0, 1), (c0*n + a1, 1)]
        )
        
        # m7 = (A[a0,b1]-A[a1,b1]) * (B[b1,c1]+B[b1,c0])
        # c01 += m7, c11 += m7
        add_term(
            [(a0*m + b1, 1), (a1*m + b1, -1)],
            [(b1*p + c1, 1), (b1*p + c0, 1)],
            [(c1*n + a0, 1), (c1*n + a1, 1)]
        )
    
    def direct_1x2x2(a_row, a_cols, b_cols):
        ar = a_row
        b0, b1 = a_cols
        c0, c1 = b_cols
        
        add_term([(ar*m + b0, 1)], [(b0*p + c0, 1)], [(c0*n + ar, 1)])
        add_term([(ar*m + b1, 1)], [(b1*p + c0, 1)], [(c0*n + ar, 1)])
        add_term([(ar*m + b0, 1)], [(b0*p + c1, 1)], [(c1*n + ar, 1)])
        add_term([(ar*m + b1, 1)], [(b1*p + c1, 1)], [(c1*n + ar, 1)])
    
    # 2x2x2 Strassen blocks (6 blocks, rank 7 each = 42)
    strassen_2x2x2([0,1], [0,1], [0,1])
    strassen_2x2x2([0,1], [0,1], [2,3])
    strassen_2x2x2([0,1], [0,1], [4,5])
    strassen_2x2x2([0,1], [2,3], [0,1])
    strassen_2x2x2([0,1], [2,3], [2,3])
    strassen_2x2x2([0,1], [2,3], [4,5])
    
    # 1x2x2 direct blocks (6 blocks, rank 4 each = 24)
    direct_1x2x2(2, [0,1], [0,1])
    direct_1x2x2(2, [0,1], [2,3])
    direct_1x2x2(2, [0,1], [4,5])
    direct_1x2x2(2, [2,3], [0,1])
    direct_1x2x2(2, [2,3], [2,3])
    direct_1x2x2(2, [2,3], [4,5])
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (idx, val) in U_terms[col]:
            U[idx][col] = val
        for (idx, val) in V_terms[col]:
            V[idx][col] = val
        for (idx, val) in W_terms[col]:
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
