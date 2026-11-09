"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    
    # Part 1: Recursive Strassen 4x4x4 (rank 49)
    # Part 2: Schoolbook 4x4x1 (rank 16)
    # Total rank: 65
    
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2
    
    # Helper: schoolbook fallback
    def schoolbook():
        U2 = np.zeros((nm, 80))
        V2 = np.zeros((mp, 80))
        W2 = np.zeros((pn, 80))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U2[i*m+j, r] = 1.0
                    V2[j*p+k, r] = 1.0
                    W2[k*n+i, r] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # A (4x4) * b (4x1)
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C
    
    # --- Part 1: Recursive Strassen 4x4x4 ---
    # Level 1: 7 Strassen products, each is a 2x2x2 multiplication
    # Level 2: Each 2x2x2 uses Strassen (7 scalar products)
    
    # Helper functions for index mapping
    def a_idx(bi, u, v):
        """Global index in A for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_idx(bi, u, v):
        """Global index in B for block bi, local (u,v) - for 4x4 part (cols 0-3)"""
        j = (bi // 2) * 2 + u  # row of B
        k = (bi % 2) * 2 + v  # col of B (0-3)
        return j * p + k
    
    def c_idx(bi, u, v):
        """Global index in C for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u  # row of C
        k = (bi % 2) * 2 + v  # col of C (0-3)
        return k * n + i
    
    # Strassen 2x2x2 definitions (for the inner level)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 outer Strassen products (Mk for k=0..6):
    # Each Mk = A'_k * B'_k where A'_k and B'_k are 2x2 matrices
    # A'_k = sum of coeff * A_block
    # B'_k = sum of coeff * B_block
    
    # Then each 2x2x2 multiplication uses inner Strassen (7 scalar products)
    
    # Outer Strassen definitions
    A_side = [
        [(0, 1.0), (3, 1.0)],   # M1: A00+A11
        [(2, 1.0), (3, 1.0)],   # M2: A10+A11
        [(0, 1.0)],             # M3: A00
        [(3, 1.0)],             # M4: A11
        [(0, 1.0), (1, 1.0)],   # M5: A00+A01
        [(2, 1.0), (0, -1.0)],  # M6: A10-A00
        [(1, 1.0), (3, -1.0)],  # M7: A01-A11
    ]
    
    B_side = [
        [(0, 1.0), (3, 1.0)],   # M1: B00+B11
        [(0, 1.0)],             # M2: B00
        [(1, 1.0), (3, -1.0)],  # M3: B01-B11
        [(2, 1.0), (0, -1.0)],  # M4: B10-B00
        [(3, 1.0)],             # M5: B11
        [(0, 1.0), (1, 1.0)],   # M6: B00+B01
        [(2, 1.0), (3, 1.0)],   # M7: B10+B11
    ]
    
    out_coeff = [
        {0: 1.0, 3: 1.0},       # M1 -> C00, C11
        {2: 1.0, 3: -1.0},      # M2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # M3 -> C01, C11
        {0: 1.0, 2: 1.0},       # M4 -> C00, C10
        {0: -1.0, 1: 1.0},      # M5 -> -C00, C01
        {3: 1.0},               # M6 -> C11
        {0: 1.0},               # M7 -> C00
    ]
    
    # Inner Strassen 2x2x2 definitions
    # For a 2x2x2 multiplication: A (2x2) * B (2x2) = C (2x2)
    # Inner Strassen products:
    # N1 = (A00+A11)(B00+B11)
    # N2 = (A10+A11)B00
    # N3 = A00(B01-B11)
    # N4 = A11(B10-B00)
    # N5 = (A00+A01)B11
    # N6 = (A10-A00)(B00+B01)
    # N7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = N1 + N4 - N5 + N7
    # C01 = N3 + N5
    # C10 = N2 + N4
    # C11 = N1 - N2 + N3 + N6
    
    inner_A_side = [
        [(0, 1.0), (3, 1.0)],   # N1: A00+A11
        [(2, 1.0), (3, 1.0)],   # N2: A10+A11
        [(0, 1.0)],             # N3: A00
        [(3, 1.0)],             # N4: A11
        [(0, 1.0), (1, 1.0)],   # N5: A00+A01
        [(2, 1.0), (0, -1.0)],  # N6: A10-A00
        [(1, 1.0), (3, -1.0)],  # N7: A01-A11
    ]
    
    inner_B_side = [
        [(0, 1.0), (3, 1.0)],   # N1: B00+B11
        [(0, 1.0)],             # N2: B00
        [(1, 1.0), (3, -1.0)],  # N3: B01-B11
        [(2, 1.0), (0, -1.0)],  # N4: B10-B00
        [(3, 1.0)],             # N5: B11
        [(0, 1.0), (1, 1.0)],   # N6: B00+B01
        [(2, 1.0), (3, 1.0)],   # N7: B10+B11
    ]
    
    inner_out_coeff = [
        {0: 1.0, 3: 1.0},       # N1 -> C00, C11
        {2: 1.0, 3: -1.0},      # N2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # N3 -> C01, C11
        {0: 1.0, 2: 1.0},       # N4 -> C00, C10
        {0: -1.0, 1: 1.0},      # N5 -> -C00, C01
        {3: 1.0},               # N6 -> C11
        {0: 1.0},               # N7 -> C00
    ]
    
    # For each outer Strassen product (mk in 0..6):
    # The 2x2 matrices A'_mk and B'_mk are formed by linear combinations
    # Then we compute A'_mk * B'_mk using inner Strassen (7 scalar products)
    
    for mk in range(7):
        # For each inner Strassen product (nk in 0..6):
        for nk in range(7):
            col = mk * 7 + nk
            
            # U: A'_mk at local position (u,v) = sum of coeff * A_block
            # A'_mk = sum_{(bi, coeff) in A_side[mk]} coeff * A_bi
            # Then inner Strassen: A''_nk = sum_{(bi2, coeff2) in inner_A_side[nk]} coeff2 * A'_mk[bi2]
            # So A''_nk[u,v] = sum_{(bi, c1) in A_side[mk]} sum_{(bi2, c2) in inner_A_side[nk]} 
            #   c1 * c2 * A_bi[local position corresponding to bi2]
            
            # Wait, I need to think about this more carefully.
            # A'_mk is a 2x2 matrix where A'_mk[i,j] = sum_{(bi, coeff) in A_side[mk]} coeff * A_bi[i,j]
            # Here i,j are local 2x2 indices.
            
            # Then inner Strassen computes A'_mk * B'_mk.
            # N_nk = A''_nk * B''_nk where A''_nk and B''_nk are linear combinations of 2x2 blocks of A'_mk and B'_mk.
            
            # But A'_mk is already a 2x2 matrix. The "blocks" of A'_mk in the inner Strassen are:
            # A'_mk[0,0], A'_mk[0,1], A'_mk[1,0], A'_mk[1,1]
            # These are scalars (elements of the 2x2 matrix).
            
            # So inner Strassen N_nk = A''_nk * B''_nk where A''_nk and B''_nk are scalars.
            # A''_nk = sum of coeff * A'_mk[bi2] for bi2 in {0,1,2,3}
            # where bi2=0 -> A'_mk[0,0], bi2=1 -> A'_mk[0,1], bi2=2 -> A'_mk[1,0], bi2=3 -> A'_mk[1,1]
            
            # A'_mk[bi2] = sum_{(bi, c1) in A_side[mk]} c1 * A_bi[local_pos(bi2)]
            # where local_pos(bi2) = (bi2//2, bi2%2)
            
            # So A''_nk = sum_{(bi, c1) in A_side[mk]} sum_{(bi2, c2) in inner_A_side[nk]} c1 * c2 * A_bi[local_pos(bi2)]
            
            # U index: global A index for block bi, local position local_pos(bi2)
            # = a_idx(bi, bi2//2, bi2%2)
            
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2
            
            # Similarly for V
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2
            
            # W: The output of inner Strassen N_nk contributes to C'_mk at certain positions
            # C'_mk = A'_mk * B'_mk
            # Inner Strassen output: C'_mk[bi_out] = sum of coeff * N_nk for each nk
            # From inner_out_coeff[nk]: {bi_out: coeff}
            # C'_mk[bi_out] is a scalar (element of the 2x2 output matrix)
            # bi_out=0 -> C'_mk[0,0], bi_out=1 -> C'_mk[0,1], bi_out=2 -> C'_mk[1,0], bi_out=3 -> C'_mk[1,1]
            
            # Then outer output: C[bi_outer] = sum of coeff * C'_mk for each mk
            # From out_coeff[mk]: {bi_outer: coeff}
            
            # So the contribution to W:
            # For each (bi_out, c_inner) in inner_out_coeff[nk]:
            #   For each (bi_outer, c_outer) in out_coeff[mk]:
            #     W[c_idx(bi_outer, bi_out//2, bi_out%2), col] += c_inner * c_outer
            
            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
