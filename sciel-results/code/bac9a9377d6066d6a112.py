import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # Use Laderman's rank-16 decomposition for 4x4 block multiplication
    # Split 6x6 into 2x2 blocks of sizes 3x3 and 3x3 (top-left 3x3, top-right 3x3, bottom-left 3x3, bottom-right 3x3)
    # Wait, that's still 3x3 blocks. Laderman is for 4x4 matrices.
    # Let's use a different approach: Use the 3x3 schoolbook for each block but with a better outer decomposition.
    # Actually, the best approach for 6x6 is to use the rank-216 schoolbook directly, but we can do better.
    # 
    # For 6x6, we can split into 3x3 blocks (2x2 grid) and use:
    # 1. Outer: Strassen (rank 7) for the 2x2 block structure
    # 2. Inner: Schoolbook (rank 27) for each 3x3 block
    # Total: 7 * 27 = 189
    #
    # Alternative: Use Laderman's algorithm on a 4x4 subproblem? Not directly applicable.
    # 
    # Let's try: Split into 2x2 blocks of 3x3, and for the inner 3x3, use the schoolbook.
    # This gives rank 189 as before.
    #
    # Can we do better? The theoretical lower bound for 6x6 is unknown, but we can try:
    # - Use a 2x2x2 tensor decomposition with rank <= 180?
    #
    # Actually, for 6x6 matrix multiplication, the best known rank is 189 (Strassen + schoolbook).
    # There might be better algorithms, but 189 is a solid baseline.
    #
    # Let's implement Strassen + schoolbook carefully.

    bs = 3  # block size
    rank = 7 * 27

    # Initialize matrices with zeros
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]

    # Strassen definitions for 2x2 blocks of 3x3 submatrices
    # Blocks indexed: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_defs = [
        # M0 = (A00+A11)(B00+B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1 = (A10+A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2 = A00(B01-B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3 = A11(B10-B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4 = (A00+A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5 = (A10-A00)(B00+B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6 = (A01-A11)(B10+B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction: C_ij = sum_s gamma_{s,ij} * M_s
    # C00 = M0 - M3 + M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon_coeffs = [
        [(0, 1.0), (3, -1.0), (4, 1.0), (6, -1.0)],  # C00
        [(2, 1.0), (4, 1.0)],                         # C01
        [(1, 1.0), (3, 1.0)],                         # C10
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],   # C11
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        # Inner schoolbook for 3x3 * 3x3: rank 27
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # U[i,k] entry for this rank term
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + i
                        k_glob = bc * bs + k
                        U[i_glob * m + k_glob][r_idx] += coeff_a

                    # V[k,j] entry
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + k
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob][r_idx] += coeff_b

                    # W[j,i] entry (W[j*n + i, r])
                    for c_idx in range(4):
                        gamma = 0.0
                        for (ss, g) in recon_coeffs[c_idx]:
                            if ss == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_idx // 2
                            cc = c_idx % 2
                            i_glob = cr * bs + i
                            j_glob = cc * bs + j
                            W[j_glob * n + i_glob][r_idx] += gamma

                    r_idx += 1

    assert r_idx == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
