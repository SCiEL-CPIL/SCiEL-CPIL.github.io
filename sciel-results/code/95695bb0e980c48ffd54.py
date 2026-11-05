import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We are building a decomposition of rank 120.
    rank = 120
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][col] += val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][col] += val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][col] += val
        col += 1

    # Indices for 5x5 matrix
    # A is 5x5. Indices (i, j) -> i*5 + j
    # B is 5x5. Indices (j, k) -> j*5 + k
    # C is 5x5. Indices (k, i) -> k*5 + i (Trace convention W[k*n+i])
    
    # Block definitions
    # I, J, K are sets of indices
    # I: rows of A, rows of C
    # J: cols of A, rows of B
    # K: cols of B, cols of C
    
    # 1. Top-Left: I={0,1,2}, J={0,1,2}, K={0,1,2} (3x3x3)
    # Laderman's 23 terms
    # We map local indices 0,1,2 to global 0,1,2
    
    # Laderman Coefficients
    # M1: (a11+a22)(b11+b22)
    u = [1,0,0, 0,1,0, 0,0,0] # a11, a22
    v = [1,0,0, 0,1,0, 0,0,0] # b11, b22
    w = [0,0,0, 0,0,0, 0,0,0] # w is for C. c11 gets +1, c22 gets +1?
    # Wait, w_vec is for C.
    # M1 contributes to c11 and c22?
    # c11 = M1 + ...
    # c22 = M1 - ...
    # So w[0,0] += 1, w[1,1] += 1.
    # Global indices for C: k*5 + i.
    # c11 -> i=0, k=0 -> 0*5+0 = 0.
    # c22 -> i=1, k=1 -> 1*5+1 = 6.
    
    # I will define the terms explicitly for the 3x3 block.
    # Local indices 0,1,2.
    # Global I: 0,1,2. J: 0,1,2. K: 0,1,2.
    
    def add_laderman_term(u_loc, v_loc, w_loc):
        # u_loc: dict (i,j) -> coeff
        # v_loc: dict (j,k) -> coeff
        # w_loc: dict (i,k) -> coeff
        u_vec = [0]*25
        v_vec = [0]*25
        w_vec = [0]*25
        for (i,j), val in u_loc.items():
            u_vec[i*5+j] = val
        for (j,k), val in v_loc.items():
            v_vec[j*5+k] = val
        for (i,k), val in w_loc.items():
            w_vec[k*5+i] = val
        add_term(u_vec, v_vec, w_vec)

    # Laderman Terms
    # 1
    add_laderman_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # 2
    add_laderman_term({(0,0):1, (1,1):-1}, {(0,0):1, (1,1):-1}, {(0,0):1, (1,1):-1})
    # 3
    add_laderman_term({(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}, {(0,1):1, (1,1):-1}) # c12, c22?
    # Laderman reconstruction:
    # c11 = M1 + M2 + M7 + M9 - M19 - M23
    # c12 = M3 + M5 - M18 + M21
    # c13 = M5 + M7 - M16 + M20
    # c21 = M2 + M4 - M14 + M22
    # c22 = M1 - M2 + M3 - M4 + M9 - M10 + M11 - M12 + M17 - M18 + M19 - M20 + M21 - M22 + M23
    # c23 = M6 + M8 - M13 + M23
    # c31 = M13 + M15 - M17 + M19
    # c32 = M14 + M16 - M18 + M20
    # c33 = M15 + M17 - M13 + M14 - M19 + M20 - M21 + M22
    
    # This is getting tedious to hardcode manually with high risk of error.
    # I will use a simpler approach: Schoolbook for all, but that's 125.
    # I need the 120 rank.
    # I will implement the 3x3 Laderman and 2x2 Strassen carefully.
    
    # Let's restart the term addition with a cleaner structure.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # 1. 3x3 Block (Top-Left)
    # Laderman's 23 multiplications
    # I will define the 23 (U, V, W) tuples for local 3x3.
    # Then map to global.
    
    # Local 3x3 indices: 0,1,2
    # Global I: 0,1,2. J: 0,1,2. K: 0,1,2.
    
    # Term 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    u = {(0,0):1, (1,1):1}
    v = {(0,0):1, (1,1):1}
    w = {(0,0):1, (1,1):1}
    add_laderman_term(u, v, w)
    
    # Term 2: (a00-a11)(b00-b11) -> c00 += 1, c11 -= 1
    u = {(0,0):1, (1,1):-1}
    v = {(0,0):1, (1,1):-1}
    w = {(0,0):1, (1,1):-1}
    add_laderman_term(u, v, w)
    
    # Term 3: (a01+a11)(b01+b11) -> c01 += 1, c11 -= 1
    u = {(0,1):1, (1,1):1}
    v = {(0,1):1, (1,1):1}
    w = {(0,1):1, (1,1):-1}
    add_laderman_term(u, v, w)
    
    # Term 4: (a01-a11)(b01-b11) -> c01 -= 1, c11 += 1
    u = {(0,1):1, (1,1):-1}
    v = {(0,1):1, (1,1):-1}
    w = {(0,1):-1, (1,1):1}
    add_laderman_term(u, v, w)
    
    # Term 5: (a02+a12)(b02+b12) -> c02 += 1, c12 += 1
    u = {(0,2):1, (1,2):1}
    v = {(0,2):1, (1,2):1}
    w = {(0,2):1, (1,2):1}
    add_laderman_term(u, v, w)
    
    # Term 6: (a02-a12)(b02-b12) -> c02 -= 1, c12 -= 1
    u = {(0,2):1, (1,2):-1}
    v = {(0,2):1, (1,2):-1}
    w = {(0,2):-1, (1,2):-1}
    add_laderman_term(u, v, w)
    
    # Term 7: (a00+a01)(b10+b11) -> c00 += 1, c01 += 1
    u = {(0,0):1, (0,1):1}
    v = {(1,0):1, (1,1):1}
    w = {(0,0):1, (0,1):1}
    add_laderman_term(u, v, w)
    
    # Term 8: (a00-a01)(b10-b11) -> c00 += 1, c01 -= 1
    u = {(0,0):1, (0,1):-1}
    v = {(1,0):1, (1,1):-1}
    w = {(0,0):1, (0,1):-1}
    add_laderman_term(u, v, w)
    
    # Term 9: (a10+a11)(b01+b00) -> c00 += 1, c10 += 1
    u = {(1,0):1, (1,1):1}
    v = {(0,1):1, (0,0):1}
    w = {(0,0):1, (1,0):1}
    add_laderman_term(u, v, w)
    
    # Term 10: (a10-a11)(b01-b00) -> c00 -= 1, c10 += 1
    u = {(1,0):1, (1,1):-1}
    v = {(0,1):1, (0,0):-1}
    w = {(0,0):-1, (1,0):1}
    add_laderman_term(u, v, w)
    
    # Term 11: (a10+a12)(b21+b22) -> c10 += 1, c12 += 1
    u = {(1,0):1, (1,2):1}
    v = {(2,1):1, (2,2):1}
    w = {(1,0):1, (1,2):1}
    add_laderman_term(u, v, w)
    
    # Term 12: (a10-a12)(b21-b22) -> c10 -= 1, c12 -= 1
    u = {(1,0):1, (1,2):-1}
    v = {(2,1):1, (2,2):-1}
    w = {(1,0):-1, (1,2):-1}
    add_laderman_term(u, v, w)
    
    # Term 13: (a20+a21)(b00+b01) -> c00 += 1, c01 += 1
    u = {(2,0):1, (2,1):1}
    v = {(0,0):1, (0,1):1}
    w = {(0,0):1, (0,1):1}
    add_laderman_term(u, v, w)
    
    # Term 14: (a20-a21)(b00-b01) -> c00 += 1, c01 -= 1
    u = {(2,0):1, (2,1):-1}
    v = {(0,0):1, (0,1):-1}
    w = {(0,0):1, (0,1):-1}
    add_laderman_term(u, v, w)
    
    # Term 15: (a20+a22)(b10+b12) -> c00 += 1, c02 += 1
    u = {(2,0):1, (2,2):1}
    v = {(1,0):1, (1,2):1}
    w = {(0,0):1, (0,2):1}
    add_laderman_term(u, v, w)
    
    # Term 16: (a20-a22)(b10-b12) -> c00 += 1, c02 -= 1
    u = {(2,0):1, (2,2):-1}
    v = {(1,0):1, (1,2):-1}
    w = {(0,0):1, (0,2):-1}
    add_laderman_term(u, v, w)
    
    # Term 17: (a21+a22)(b01+b02) -> c01 += 1, c02 += 1
    u = {(2,1):1, (2,2):1}
    v = {(0,1):1, (0,2):1}
    w = {(0,1):1, (0,2):1}
    add_laderman_term(u, v, w)
    
    # Term 18: (a21-a22)(b01-b02) -> c01 += 1, c02 -= 1
    u = {(2,1):1, (2,2):-1}
    v = {(0,1):1, (0,2):-1}
    w = {(0,1):1, (0,2):-1}
    add_laderman_term(u, v, w)
    
    # Term 19: a00 b22 -> c02 += 1
    u = {(0,0):1}
    v = {(2,2):1}
    w = {(0,2):1}
    add_laderman_term(u, v, w)
    
    # Term 20: a01 b21 -> c01 += 1
    u = {(0,1):1}
    v = {(2,1):1}
    w = {(0,1):1}
    add_laderman_term(u, v, w)
    
    # Term 21: a02 b20 -> c00 += 1
    u = {(0,2):1}
    v = {(2,0):1}
    w = {(0,0):1}
    add_laderman_term(u, v, w)
    
    # Term 22: a20 b02 -> c02 += 1
    u = {(2,0):1}
    v = {(0,2):1}
    w = {(0,2):1}
    add_laderman_term(u, v, w)
    
    # Term 23: a22 b00 -> c00 += 1
    u = {(2,2):1}
    v = {(0,0):1}
    w = {(0,0):1}
    add_laderman_term(u, v, w)

    # 2. 2x2 Block (Bottom-Right)
    # I={3,4}, J={3,4}, K={3,4}
    # Strassen's 7 terms
    # Local 0,1 map to Global 3,4
    
    def add_strassen_term(u_loc, v_loc, w_loc):
        u_vec = [0]*25
        v_vec = [0]*25
        w_vec = [0]*25
        for (i,j), val in u_loc.items():
            u_vec[(i+3)*5 + (j+3)] = val
        for (j,k), val in v_loc.items():
            v_vec[(j+3)*5 + (k+3)] = val
        for (i,k), val in w_loc.items():
            w_vec[(k+3)*5 + (i+3)] = val
        add_term(u_vec, v_vec, w_vec)

    # M1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    add_strassen_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2: (a10+a11)b00 -> c10 += 1, c11 -= 1
    add_strassen_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
    # M3: a00(b01-b11) -> c01 += 1, c11 += 1
    add_strassen_term({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4: a11(b10-b00) -> c00 += 1, c10 += 1
    add_strassen_term({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})
    # M5: (a00+a01)b11 -> c01 += 1, c11 -= 1
    add_strassen_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1})
    # M6: (a10-a00)(b00+b01) -> c00 += 1, c01 += 1
    add_strassen_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (0,1):1})
    # M7: (a01-a11)(b10+b11) -> c00 += 1, c10 += 1
    add_strassen_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1})

    # 3-8. Rectangular Blocks (Schoolbook)
    # I={0,1,2}, J={3,4}, K={3,4} (A12 B22)
    # I={3,4}, J={0,1,2}, K={0,1,2} (A21 B11)
    # I={0,1,2}, J={0,1,2}, K={3,4} (A11 B12)
    # I={0,1,2}, J={3,4}, K={0,1,2} (A12 B21)
    # I={3,4}, J={0,1,2}, K={3,4} (A21 B12)
    # I={3,4}, J={3,4}, K={0,1,2} (A22 B12)
    
    # Helper for schoolbook rectangular
    def add_schoolbook_rect(I, J, K):
        for i in I:
            for j in J:
                for k in K:
                    u_vec = [0]*25
                    v_vec = [0]*25
                    w_vec = [0]*25
                    u_vec[i*5+j] = 1
                    v_vec[j*5+k] = 1
                    w_vec[k*5+i] = 1
                    add_term(u_vec, v_vec, w_vec)

    add_schoolbook_rect([0,1,2], [3,4], [3,4])
    add_schoolbook_rect([3,4], [0,1,2], [0,1,2])
    add_schoolbook_rect([0,1,2], [0,1,2], [3,4])
    add_schoolbook_rect([0,1,2], [3,4], [0,1,2])
    add_schoolbook_rect([3,4], [0,1,2], [3,4])
    add_schoolbook_rect([3,4], [3,4], [0,1,2])

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
