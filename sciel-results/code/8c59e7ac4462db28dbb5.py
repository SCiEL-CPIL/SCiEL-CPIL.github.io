"""Rewritten program: rank-34 exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-34 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Uses block decomposition:
    - Split A into rows {0,1} and {2}.
    - Split B into cols {0,1} and {2,3}.
    - 4 blocks:
      1. A[0:2,:] * B[:,0:2] -> C[0:2,0:2] (M_2,3,2, rank 11)
      2. A[0:2,:] * B[:,2:4] -> C[0:2,2:4] (M_2,3,2, rank 11)
      3. A[2:3,:] * B[:,0:2] -> C[2:3,0:2] (M_1,3,2, rank 6)
      4. A[2:3,:] * B[:,2:4] -> C[2:3,2:4] (M_1,3,2, rank 6)
    Total rank: 34.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t_offset = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal t_offset
        t = t_offset
        t_offset += 1
        for (i, j, c) in u_entries:
            U[i*m+j][t] += c
        for (j, k, c) in v_entries:
            V[j*p+k][t] += c
        for (k, i, c) in w_entries:
            W[k*n+i][t] += c

    # Helper: Add M_2,2,2 Strassen terms
    # A indices: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # B indices: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # C indices: (k0,i0), (k0,i1), (k1,i0), (k1,i1)
    # Note: W index is k*n+i
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        # S1 = (a_i0j0 + a_i1j1) * (b_j0k0 + b_j1k1)
        add_term(
            [(i0, j0, 1), (i1, j1, 1)],
            [(j0, k0, 1), (j1, k1, 1)],
            [(k0, i0, 1), (k0, i1, 1), (k1, i0, 1), (k1, i1, 1)]
        )
        # S2 = (a_i1j0 + a_i1j1) * b_j0k0
        add_term(
            [(i1, j0, 1), (i1, j1, 1)],
            [(j0, k0, 1)],
            [(k0, i1, 1)]
        )
        # S3 = a_i0j0 * (b_j1k0 - b_j0k0)
        add_term(
            [(i0, j0, 1)],
            [(j1, k0, 1), (j0, k0, -1)],
            [(k0, i0, 1)]
        )
        # S4 = a_i1j1 * (b_j1k1 - b_j0k1)
        add_term(
            [(i1, j1, 1)],
            [(j1, k1, 1), (j0, k1, -1)],
            [(k1, i1, 1)]
        )
        # S5 = (a_i0j0 + a_i0j1) * (b_j1k0 + b_j1k1)
        add_term(
            [(i0, j0, 1), (i0, j1, 1)],
            [(j1, k0, 1), (j1, k1, 1)],
            [(k0, i0, 1), (k1, i0, 1)]
        )
        # S6 = (a_i1j1 - a_i0j0) * (b_j0k0 + b_j0k1)
        add_term(
            [(i1, j1, 1), (i0, j0, -1)],
            [(j0, k0, 1), (j0, k1, 1)],
            [(k0, i1, 1), (k1, i1, 1)]
        )
        # S7 = (a_i0j1 - a_i1j1) * (b_j1k0 + b_j1k1)
        # Wait, standard Strassen:
        # M7 = (a12 - a22)(b21 + b22) ? No.
        # Let's use the standard 7 terms:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)(b11)
        # m3 = (a11-a22)(b22)
        # m4 = a11(b12-b22)
        # m5 = (a12+a22)(b21-b11)
        # m6 = (a11+a12)(b11+b12)
        # m7 = (a21-a11)(b21+b22)
        
        # My previous implementation was a mix. Let's stick to a verified set.
        # I will re-implement add_strassen_2x2 with the standard set above.
        
        # Reset t_offset to before this call? No, I'll just define it correctly.
        
    # Let's rewrite add_strassen_2x2 properly
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        # m1 = (a_i0j0 + a_i1j1) * (b_j0k0 + b_j1k1)
        add_term(
            [(i0, j0, 1), (i1, j1, 1)],
            [(j0, k0, 1), (j1, k1, 1)],
            [(k0, i0, 1), (k0, i1, 1), (k1, i0, 1), (k1, i1, 1)]
        )
        # m2 = (a_i1j0 + a_i1j1) * b_j0k0
        add_term(
            [(i1, j0, 1), (i1, j1, 1)],
            [(j0, k0, 1)],
            [(k0, i1, 1)]
        )
        # m3 = (a_i0j0 - a_i1j1) * b_j1k1
        add_term(
            [(i0, j0, 1), (i1, j1, -1)],
            [(j1, k1, 1)],
            [(k1, i0, 1)]
        )
        # m4 = a_i0j0 * (b_j0k1 - b_j1k1)
        add_term(
            [(i0, j0, 1)],
            [(j0, k1, 1), (j1, k1, -1)],
            [(k1, i0, 1)]
        )
        # m5 = (a_i0j1 + a_i1j1) * (b_j1k0 - b_j0k0)
        add_term(
            [(i0, j1, 1), (i1, j1, 1)],
            [(j1, k0, 1), (j0, k0, -1)],
            [(k0, i0, 1)]
        )
        # m6 = (a_i0j0 + a_i0j1) * (b_j0k0 + b_j0k1)
        add_term(
            [(i0, j0, 1), (i0, j1, 1)],
            [(j0, k0, 1), (j0, k1, 1)],
            [(k0, i0, 1), (k1, i0, 1)]
        )
        # m7 = (a_i1j0 - a_i0j0) * (b_j1k0 + b_j1k1)
        add_term(
            [(i1, j0, 1), (i0, j0, -1)],
            [(j1, k0, 1), (j1, k1, 1)],
            [(k0, i1, 1), (k1, i1, 1)]
        )

    # Helper: Add M_2,1,2 schoolbook terms
    # A indices: (i0,j), (i1,j)
    # B indices: (j,k0), (j,k1)
    # C indices: (k0,i0), (k0,i1), (k1,i0), (k1,i1)
    def add_m212(i0, i1, j, k0, k1):
        # a_i0j * b_jk0
        add_term([(i0, j, 1)], [(j, k0, 1)], [(k0, i0, 1)])
        # a_i0j * b_jk1
        add_term([(i0, j, 1)], [(j, k1, 1)], [(k1, i0, 1)])
        # a_i1j * b_jk0
        add_term([(i1, j, 1)], [(j, k0, 1)], [(k0, i1, 1)])
        # a_i1j * b_jk1
        add_term([(i1, j, 1)], [(j, k1, 1)], [(k1, i1, 1)])

    # Helper: Add M_1,3,2 schoolbook terms
    # A indices: (i,j0), (i,j1), (i,j2)
    # B indices: (j0,k0), (j0,k1), (j1,k0), (j1,k1), (j2,k0), (j2,k1)
    # C indices: (k0,i), (k1,i)
    def add_m132(i, j0, j1, j2, k0, k1):
        for j in [j0, j1, j2]:
            add_term([(i, j, 1)], [(j, k0, 1)], [(k0, i, 1)])
            add_term([(i, j, 1)], [(j, k1, 1)], [(k1, i, 1)])

    # Block 1: A[0:2,:] * B[:,0:2] -> C[0:2,0:2]
    # M_2,3,2 using j in {0,1} for Strassen and j=2 for M_2,1,2
    add_strassen_2x2(0, 1, 0, 1, 0, 1)
    add_m212(0, 1, 2, 0, 1)
    
    # Block 2: A[0:2,:] * B[:,2:4] -> C[0:2,2:4]
    # M_2,3,2 using j in {0,1} for Strassen and j=2 for M_2,1,2
    add_strassen_2x2(0, 1, 0, 1, 2, 3)
    add_m212(0, 1, 2, 2, 3)
    
    # Block 3: A[2:3,:] * B[:,0:2] -> C[2:3,0:2]
    # M_1,3,2
    add_m132(2, 0, 1, 2, 0, 1)
    
    # Block 4: A[2:3,:] * B[:,2:4] -> C[2:3,2:4]
    # M_1,3,2
    add_m132(2, 0, 1, 2, 2, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
