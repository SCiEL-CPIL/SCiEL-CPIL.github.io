"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 17 decomposition using Strassen on 2x2 sub-block
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 component
    # u_idx: list of (index, value)
    # v_idx: list of (index, value)
    # w_idx: list of (index, value)
    def set_comp(col, u_idx, v_idx, w_idx):
        for idx, val in u_idx:
            U[idx][col] = val
        for idx, val in v_idx:
            V[idx][col] = val
        for idx, val in w_idx:
            W[idx][col] = val

    # Indices for A (2x3):
    # a11=0, a12=1, a13=2
    # a21=3, a22=4, a23=5
    
    # Indices for B (3x3):
    # b11=0, b12=1, b13=2
    # b21=3, b22=4, b23=5
    # b31=6, b32=7, b33=8
    
    # Indices for C (2x3):
    # c11=0, c12=2, c13=4
    # c21=1, c22=3, c23=5
    
    # 1. M1 = (a11 + a22)(b11 + b22)
    set_comp(0, [(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)])
    
    # 2. M2 = (a21 + a22)b11
    set_comp(1, [(3, 1), (4, 1)], [(0, 1)], [(1, 1)])
    
    # 3. M3 = a11(b12 - b22)
    set_comp(2, [(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)])
    
    # 4. M4 = a22(b21 - b11)
    set_comp(3, [(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # 5. M5 = (a11 + a12)b22
    set_comp(4, [(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)])
    
    # 6. M6 = (a21 - a11)(b11 + b12)
    set_comp(5, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # 7. M7 = (a12 - a22)(b21 + b22)
    set_comp(6, [(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)])
    
    # 8. M8 = a13 b31
    set_comp(7, [(2, 1)], [(6, 1)], [(0, 1)])
    
    # 9. M9 = a13 b32
    set_comp(8, [(2, 1)], [(7, 1)], [(2, 1)])
    
    # 10. M10 = a23 b31
    set_comp(9, [(5, 1)], [(6, 1)], [(1, 1)])
    
    # 11. M11 = a23 b32
    set_comp(10, [(5, 1)], [(7, 1)], [(3, 1)])
    
    # 12. M12 = a11 b13
    set_comp(11, [(0, 1)], [(2, 1)], [(4, 1)])
    
    # 13. M13 = a12 b23
    set_comp(12, [(1, 1)], [(5, 1)], [(4, 1)])
    
    # 14. M14 = a21 b13
    set_comp(13, [(3, 1)], [(2, 1)], [(5, 1)])
    
    # 15. M15 = a22 b23
    set_comp(14, [(4, 1)], [(5, 1)], [(5, 1)])
    
    # 16. M16 = a13 b33
    set_comp(15, [(2, 1)], [(8, 1)], [(4, 1)])
    
    # 17. M17 = a23 b33
    set_comp(16, [(5, 1)], [(8, 1)], [(5, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
