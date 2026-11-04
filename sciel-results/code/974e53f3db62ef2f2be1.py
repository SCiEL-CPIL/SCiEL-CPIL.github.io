"""
Matrix multiplication 4x5 by 5x6 — rank-minimal exact tensor decomposition.

Strategy:
  1. Partition n=4 into 2+2, m=5 into 2+2+1, p=6 into 3+3.
  2. Each of the 2*3*3 = 18 blocks is a small (n_i, m_j, p_k) multiplication.
  3. Apply the rank-7 Strassen decomposition to the four 2x2x2 blocks.
  4. Use schoolbook for the remaining 14 rectangular blocks.
  5. Total rank = 4*7 + 14*schoolbook = 28 + (16+24+48+16+24+24) = 28 + 156 = 184?
     Wait, that's worse than 120. Let me recalculate.

     Schoolbook ranks for each block:
     (2,2,2): 8  -> Strassen: 7   (4 blocks)
     (2,2,3): 12
     (2,3,2): 12
     (2,3,3): 18
     (2,2,2): 8  -> Strassen: 7   (already counted)
     Actually, the 18 blocks are:
     n_i in {2,2}, m_j in {2,2,1}, p_k in {3,3}
     So the blocks are:
     (2,2,3) x4, (2,2,3) x4, (2,1,3) x4, (2,2,3) x4, (2,2,3) x4, (2,1,3) x4
     Wait, let me enumerate:
     i_block: 0 (n=2), 1 (n=2)
     j_block: 0 (m=2), 1 (m=2), 2 (m=1)
     k_block: 0 (p=3), 1 (p=3)
     
     Blocks:
     (2,2,3): i=0,j=0,k=0; i=0,j=1,k=0; i=1,j=0,k=0; i=1,j=1,k=0 -> 4 blocks, rank 12 each
     (2,2,3): i=0,j=0,k=1; i=0,j=1,k=1; i=1,j=0,k=1; i=1,j=1,k=1 -> 4 blocks, rank 12 each
     (2,1,3): i=0,j=2,k=0; i=1,j=2,k=0 -> 2 blocks, rank 6 each
     (2,1,3): i=0,j=2,k=1; i=1,j=2,k=1 -> 2 blocks, rank 6 each
     
     Total: 8*12 + 4*6 = 96 + 24 = 120. That's the same as schoolbook!

     So the Strassen trick doesn't help here because the 2x2x2 blocks don't appear
     in this particular partition (p is 3, not 2).

     Let me try a different partition: n=4 as 2+2, m=5 as 2+3, p=6 as 3+3.
     Blocks: (2,2,3), (2,2,3), (2,3,3), (2,3,3) x2 each for k=0,1
     (2,2,3): 4 blocks, rank 12
     (2,3,3): 4 blocks, rank 18
     Total: 4*12 + 4*18 = 48 + 72 = 120. Still 120.

     What about n=4 as 1+3, m=5 as 2+3, p=6 as 3+3?
     (1,2,3): 2 blocks, rank 6
     (1,3,3): 2 blocks, rank 9
     (3,2,3): 2 blocks, rank 18
     (3,3,3): 2 blocks, rank 27
     Total: 2*(6+9+18+27) = 2*60 = 120.

     It seems like any partition gives exactly 120 when using schoolbook for each block,
     because the total number of (i,j,k) triples is always 4*5*6 = 120.

     To get below 120, we MUST use a non-schoolbook decomposition for at least one block.
     The only standard non-schoolbook decomposition I know is Strassen for 2x2x2 (rank 7 vs 8).

     So I need a partition where at least one block is 2x2x2.
     Let's try: n=4 as 2+2, m=5 as 2+2+1, p=6 as 2+2+2.
     Blocks: 2 * 3 * 3 = 18 blocks.
     (2,2,2): 4 blocks -> Strassen rank 7 each -> 28
     (2,2,2): 4 blocks -> Strassen rank 7 each -> 28
     Wait, m has 3 parts (2,2,1) and p has 3 parts (2,2,2).
     (2,2,2): i=0,1; j=0,1; k=0,1 -> 4 blocks, Strassen 7 each -> 28
     (2,2,2): i=0,1; j=0,1; k=2 -> 4 blocks, Strassen 7 each -> 28
     (2,2,2): i=0,1; j=2; k=0,1 -> 2*1*2 = 4 blocks? No, j=2 has m=1.
     
     Let me be precise:
     n parts: [2, 2]
     m parts: [2, 2, 1]
     p parts: [2, 2, 2]
     
     Blocks:
     (2,2,2): 2*2*3 = 12 blocks, Strassen 7 each -> 84
     (2,2,2): already counted
     (2,1,2): 2*1*3 = 6 blocks, schoolbook 4 each -> 24
     Total: 84 + 24 = 108.
     
     That's better than 120!
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Partition
    n_parts = [2, 2]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]

    # Compute block offsets
    n_offsets = [0]
    for s in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + s)

    # Collect all blocks with their (n_i, m_j, p_k) sizes
    blocks = []
    for ii, ni in enumerate(n_parts):
        for jj, mj in enumerate(m_parts):
            for kk, pk in enumerate(p_parts):
                blocks.append((ni, mj, pk, n_offsets[ii], m_offsets[jj], p_offsets[kk]))

    # For each block, determine decomposition
    # Strassen 2x2x2: rank 7
    # Schoolbook: rank n*m*p

    strassen_entries = []  # list of (U_entries, V_entries, W_entries) for 2x2x2
    # Strassen for 2x2: 
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b21)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6

    # For each 2x2x2 block at offset (no, mo, po):
    # The tensor entries are T[no + i*2 + j, mo + j*2 + k, po + k*2 + i] = 1
    # for i,j,k in {0,1}
    # In terms of the global indices:
    # U index: no + i*2 + j  (i is row of A, j is col of A / row of B)
    # V index: mo + j*2 + k  (j is row of B, k is col of B)
    # W index: po + k*2 + i  (k is col of B, i is row of C)
    # 
    # Strassen terms for this block:
    # m1: (a11+a22)(b11+b22)
    #   U: entries at (i=0,j=0) and (i=1,j=1) -> U[no+0][r] += 1, U[no+1*2+1][r] += 1
    #        Actually U index is no + i*2 + j. For (i=0,j=0): no+0. For (i=1,j=1): no+3.
    #   V: entries at (j=0,k=0) and (j=1,k=1) -> V[mo+0][r] += 1, V[mo+1*2+1][r] += 1
    #   W: entries at (k=0,i=0) and (k=1,i=1) -> W[po+0][r] += 1, W[po+1*2+1][r] += 1
    #   So U has +1 at no+0 and no+3, V has +1 at mo+0 and mo+3, W has +1 at po+0 and po+3.

    # Let me define Strassen terms more carefully.
    # For a 2x2x2 block at offset (no, mo, po), the Strassen decomposition uses 7 terms.
    # Each term r has a set of (U_indices, V_indices, W_indices, coefficient).

    def strassen_2x2x2(no, mo, po):
        """Return list of 7 (u_entries, v_entries, w_entries) where each is list of (index, coeff)."""
        terms = []
        # m1 = (a11+a22)(b11+b22)
        u1 = [(no + 0, 1), (no + 3, 1)]  # i=0,j=0 and i=1,j=1
        v1 = [(mo + 0, 1), (mo + 3, 1)]  # j=0,k=0 and j=1,k=1
        w1 = [(po + 0, 1), (po + 3, 1)]  # k=0,i=0 and k=1,i=1
        terms.append((u1, v1, w1))

        # m2 = (a21+a22)b11
        u2 = [(no + 2, 1), (no + 3, 1)]  # i=1,j=0 and i=1,j=1
        v2 = [(mo + 0, 1)]  # j=0,k=0
        w2 = [(po + 0, 1)]  # k=0,i=0
        terms.append((u2, v2, w2))

        # m3 = a11(b12-b22)
        u3 = [(no + 0, 1)]  # i=0,j=0
        v3 = [(mo + 1, 1), (mo + 3, -1)]  # j=0,k=1 and j=1,k=1
        w3 = [(po + 1, 1)]  # k=1,i=0
        terms.append((u3, v3, w3))

        # m4 = a22(b21-b11)
        u4 = [(no + 3, 1)]  # i=1,j=1
        v4 = [(mo + 2, 1), (mo + 0, -1)]  # j=1,k=0 and j=0,k=0
        w4 = [(po + 2, 1)]  # k=0,i=1
        terms.append((u4, v4, w4))

        # m5 = (a11+a12)b22
        u5 = [(no + 0, 1), (no + 1, 1)]  # i=0,j=0 and i=0,j=1
        v5 = [(mo + 3, 1)]  # j=1,k=1
        w5 = [(po + 3, 1)]  # k=1,i=1
        terms.append((u5, v5, w5))

        # m6 = (a21-a11)(b11+b21)
        u6 = [(no + 2, 1), (no + 0, -1)]  # i=1,j=0 and i=0,j=0
        v6 = [(mo + 0, 1), (mo + 2, 1)]  # j=0,k=0 and j=1,k=0
        w6 = [(po + 0, 1)]  # k=0,i=0
        terms.append((u6, v6, w6))

        # m7 = (a12-a22)(b21+b22)
        u7 = [(no + 1, 1), (no + 3, -1)]  # i=0,j=1 and i=1,j=1
        v7 = [(mo + 2, 1), (mo + 3, 1)]  # j=1,k=0 and j=1,k=1
        w7 = [(po + 3, 1)]  # k=1,i=1
        terms.append((u7, v7, w7))

        return terms

    # Build the decomposition
    U_list = []  # will be list of (index, value) per column
    V_list = []
    W_list = []

    col = 0
    for (ni, mj, pk, no, mo, po) in blocks:
        if ni == 2 and mj == 2 and pk == 2:
            # Use Strassen
            for (u_entries, v_entries, w_entries) in strassen_2x2x2(no, mo, po):
                U_list.append([(no + i*2 + j, c) for (idx, c) in u_entries])
                # Actually u_entries are already (global_index, coeff)
                U_list.append(u_entries)
                V_list.append(v_entries)
                W_list.append(w_entries)
                col += 1
        else:
            # Schoolbook
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        u_idx = no + i * mj + j
                        v_idx = mo + j * pk + k
                        w_idx = po + k * ni + i
                        U_list.append([(u_idx, 1)])
                        V_list.append([(v_idx, 1)])
                        W_list.append([(w_idx, 1)])
                        col += 1

    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for (idx, val) in U_list[r]:
            U[idx][r] += val
        for (idx, val) in V_list[r]:
            V[idx][r] += val
        for (idx, val) in W_list[r]:
            W[idx][r] += val

    return {"U": U, "V": V, "W": W}
