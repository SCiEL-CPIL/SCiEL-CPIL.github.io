"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # Strategy: Divide-and-conquer using Strassen on 2x2 sub-blocks.
    # Decompose M_{3,3,8} by splitting the 3x3 matrix A into:
    #   A = [[A22, A21], [A12, A11]]  (2x2, 2x1, 1x2, 1x1 blocks)
    # and B (3x8) into:
    #   B = [[B2], [B1]]  (2x8, 1x8 blocks)
    #
    # C = A*B gives:
    #   C_top = A22*B2 + A21*B1  (2x8)
    #   C_bot = A12*B2 + A11*B1  (1x8)
    #
    # For the 2x3x8 part (C_top): multiply 2x3 by 3x8.
    #   Split into 2x2x8 (Strassen-able) + 2x1x8 (rank 16)
    #   2x2x8 = 2x2x2 (rank 7) tensor 1x1x4 (rank 4) = rank 28
    #   2x1x8 = rank 16
    #   Total for C_top: 28 + 16 = 44... 
    #
    # Actually, let me use a cleaner decomposition:
    # M_{3,3,8} = M_{2,3,8} + M_{1,3,8}
    # M_{2,3,8}: multiply 2x3 by 3x8 -> 2x8 result
    #   = M_{2,2,8} + M_{2,1,8}
    #   M_{2,2,8} = M_{2,2,2} ⊗ M_{1,1,4}: rank 7*4 = 28
    #   M_{2,1,8}: rank 2*8 = 16
    #   M_{2,3,8} rank <= 44
    # M_{1,3,8}: multiply 1x3 by 3x8 -> 1x8 result, rank 24
    # Total: 44 + 24 = 68
    #
    # Let's implement this rank-68 construction.
    
    U = []  # will be list of lists, shape (n*m, rank)
    V = []  # shape (m*p, rank)
    W = []  # shape (p*n, rank)
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 term."""
        U.append([0] * (len(U) + 1) if U else [u_val])
        V.append([0] * (len(V) + 1) if V else [v_val])
        W.append([0] * (len(W) + 1) if W else [w_val])
        # Extend all existing rows
        for i in range(len(U)):
            if i < len(U) - 1:
                U[i].append(0)
        for i in range(len(V)):
            if i < len(V) - 1:
                V[i].append(0)
        for i in range(len(W)):
            if i < len(W) - 1:
                W[i].append(0)
        U[-1][-1] = u_val
        V[-1][-1] = v_val
        W[-1][-1] = w_val
    
    # This approach is getting complex. Let me use a direct construction.
    # 
    # Rank-68 construction:
    # Part 1: M_{2,2,2} Strassen (rank 7) tensor M_{1,1,4} (rank 4) = 28 terms
    #   Handles A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # Part 2: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]: rank 2*8=16 terms
    # Part 3: A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]: rank 2*8=16 terms  
    # Part 4: A[2, 2] * B[2, 0:8] -> C[2, 0:8]: rank 1*8=8 terms
    # Total: 28 + 16 + 16 + 8 = 68
    
    # Strassen 2x2x2 decomposition (rank 7):
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    # For M_{2,2,2}, the U,V,W for each Strassen term:
    # Term r: U has entries for A indices, V for B indices, W for C indices
    
    # Let me build this more carefully using arrays.
    
    total_rank = 68
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen 2x2x2 tensor 1x1x4 (28 terms)
    # A indices: i in {0,1}, j in {0,1} -> U index i*m+j = i*3+j
    # B indices: j in {0,1}, k in {0..7} -> V index j*p+k = j*8+k
    # C indices: i in {0,1}, k in {0..7} -> W index k*n+i = k*3+i
    
    # Strassen multipliers for 2x2:
    strassen_U = [
        # Each entry is a dict mapping (i,j) -> coeff
        { (0,0): 1, (1,1): 1 },   # m1: (a00+a11)
        { (1,0): 1, (1,1): 1 },   # m2: (a10+a11)
        { (0,0): 1 },             # m3: a00
        { (1,1): 1 },             # m4: a11
        { (0,0): 1, (0,1): 1 },   # m5: (a00+a01)
        { (1,0): 1, (0,0): -1 },  # m6: (a10-a00)
        { (0,1): 1, (1,1): -1 },  # m7: (a01-a11)
    ]
    strassen_V = [
        { (0,0): 1, (1,1): 1 },   # (b00+b11)
        { (0,0): 1 },             # b00
        { (0,1): 1, (1,1): -1 },  # (b01-b11)
        { (1,0): 1, (0,0): -1 },  # (b10-b00)
        { (1,1): 1 },             # b11
        { (0,0): 1, (0,1): 1 },   # (b00+b01)
        { (1,0): 1, (1,1): 1 },   # (b10+b11)
    ]
    strassen_W = [
        { (0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1 },   # c00 = m1+m4-m5+m7
        { (0,1): 1, (1,1): 1 },                         # c01 = m3+m5
        { (1,0): 1, (1,1): 1 },                         # c10 = m2+m4
        { (0,0): 1, (1,0): -1, (0,1): 1, (1,1): 1 },   # c11 = m1-m2+m3+m6
    ]
    # Wait, the W coefficients are wrong. Let me redo.
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # For the tensor product with 1x1x4 (columns 0-3 of B/C):
    # Each Strassen term r is multiplied by each column offset t in 0..3
    
    for r in range(7):
        for t in range(4):
            # U part
            for (ii, jj), coeff in strassen_U[r].items():
                U[ii * m + jj][col] += coeff
            # V part: B indices (j, k) where j in {0,1}, k = t
            for (jj, kk), coeff in strassen_V[r].items():
                V[jj * p + kk * 2 + t][col] += coeff  # WRONG: k should be 0..7
            col += 1
    
    # This is getting error-prone. Let me use a completely different, cleaner approach.
    # 
    # I'll directly construct the rank-68 decomposition using explicit index tracking.
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Strassen 2x2x2 ⊗ 1x1x4 for A[0:2,0:2]*B[0:2,0:8]->C[0:2,0:8]
    # 7 Strassen multipliers × 4 column-groups = 28 terms
    # For column group t (t=0,1,2,3), columns are k=2t and k=2t+1... 
    # No wait, 1x1x4 means we just replicate for 4 independent column positions.
    # Actually M_{2,2,8} = M_{2,2,2} ⊗ M_{1,1,4} means:
    # For each of the 4 "blocks" of 2 columns, apply Strassen.
    # Block t uses B columns {2t, 2t+1} and produces C columns {2t, 2t+1}.
    
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        for r in range(7):
            # U: A indices
            for (ii, jj), coeff in strassen_U[r].items():
                U[ii * m + jj][col] += coeff
            # V: B indices (j, k) where j in {0,1}, k in {k0, k1}
            # Map local (0,0)->(j=0,k=k0), (0,1)->(j=0,k=k1), (1,0)->(j=1,k=k0), (1,1)->(j=1,k=k1)
            for (jj_local, kk_local), coeff in strassen_V[r].items():
                jj = jj_local
                kk = k0 + kk_local
                V[jj * p + kk][col] += coeff
            # W: C indices (i, k) where i in {0,1}, k in {k0, k1}
            # c00->(i=0,k=k0), c01->(i=0,k=k1), c10->(i=1,k=k0), c11->(i=1,k=k1)
            w_coeffs = [
                [(0, k0, 1), (1, k1, 1), (0, k1, -1), (1, k0, 1)],   # m1: c00+m11-c01+m10
                [(0, k1, 1), (1, k1, 1)],                              # m3: c01
                [(1, k0, 1), (1, k1, 1)],                              # m2: c10
                [(0, k0, 1), (1, k0, -1), (0, k1, 1), (1, k1, 1)],    # m4: c00-c10+c01+c11
            ]
            # This mapping is wrong. Let me use the correct Strassen output formulas.
            # c00 = m1 + m4 - m5 + m7
            # c01 = m3 + m5  
            # c10 = m2 + m4
            # c11 = m1 - m2 + m3 + m6
            # So for multiplier r, which C entries does it contribute to?
            # r=0 (m1): c00(+1), c11(+1)
            # r=1 (m2): c10(+1), c11(-1)
            # r=2 (m3): c01(+1), c11(+1)
            # r=3 (m4): c00(+1), c10(+1)
            # r=4 (m5): c00(-1), c01(+1)
            # r=5 (m6): c11(+1)
            # r=6 (m7): c00(+1)
            
            w_map = [
                [(0, k0, 1), (1, k1, 1)],    # m1 -> c00, c11
                [(1, k0, 1), (1, k1, -1)],   # m2 -> c10, c11
                [(0, k1, 1), (1, k1, 1)],    # m3 -> c01, c11
                [(0, k0, 1), (1, k0, 1)],    # m4 -> c00, c10
                [(0, k0, -1), (0, k1, 1)],   # m5 -> c00, c01
                [(1, k1, 1)],                # m6 -> c11
                [(0, k0, 1)],                # m7 -> c00
            ]
            for (ii, kk, coeff) in w_map[r]:
                W[kk * n + ii][col] += coeff
            col += 1
    
    # Part 2: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]: 2*8=16 terms
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    # Part 3: A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]: 2*8=16 terms
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 2][col] = 1.0
            col += 1
    
    # Part 4: A[2, 2] * B[2, 0:8] -> C[2, 0:8]: 1*8=8 terms
    for k in range(8):
        U[2 * m + 2][col] = 1.0
        V[2 * p + k][col] = 1.0
        W[k * n + 2][col] = 1.0
        col += 1
    
    assert col == total_rank, f"Expected {total_rank} terms, got {col}"
    
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
