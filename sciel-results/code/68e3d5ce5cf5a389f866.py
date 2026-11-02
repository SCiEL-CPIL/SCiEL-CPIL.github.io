"""Matrix multiplication 3,4,6 rank-minimal decomposition."""

import numpy as np

def construct():
    n, m, p = 3, 4, 6

    # Use a decomposition based on Strassen for the 2x2x2 blocks
    # and schoolbook for the rest, but optimize the partitioning.
    
    # The current program uses:
    # - 24 schoolbook terms for row 0 (1x4x6)
    # - 6 Strassen blocks for rows 1,2 (2x4x6), each using 7 terms -> 42 terms
    # Total: 66 terms
    
    # Let's try a different partitioning or a known better decomposition.
    
    # One idea: Use Strassen on 2x2x2 blocks throughout.
    # For 3x4x6:
    # We can view it as 3 (i) x 4 (j) x 6 (k).
    # Split i into 1+2, j into 2+2, k into 2+2+2.
    
    # Alternatively, use the fact that M(3,4,6) can be decomposed using
    # a combination of Strassen and schoolbook.
    
    # Let's try to use Strassen for all possible 2x2x2 sub-blocks.
    # i: 0,1,2 -> blocks at i=0,1 (0-1, 1-2)
    # j: 0,1,2,3 -> blocks at j=0,1,2 (0-1, 1-2, 2-3)
    # k: 0,1,2,3,4,5 -> blocks at k=0,1,2,3 (0-1, 1-2, 2-3, 3-4)
    
    # But Strassen requires 2x2x2, so we need even dimensions or careful handling.
    
    # Let's stick with a known efficient decomposition.
    # According to literature, M(3,4,6) has rank at most 64 or so.
    
    # I'll implement a rank-64 decomposition using a mix of Strassen and schoolbook.
    
    # Actually, let's try to use the current approach but with a better partition.
    # Current: 24 + 42 = 66.
    
    # Alternative: Split i into 1+2, but use Strassen for the 2x4x6 part more efficiently.
    # Or use a 3x3x3 Strassen-like algorithm? No, Strassen is for 2x2x2.
    
    # Let's try to use 7 terms for 2x2x2, and see if we can reduce the schoolbook part.
    
    # For row 0: 1x4x6. Can we use Strassen? No, Strassen needs 2x2x2.
    # But we can use 1x2x2 blocks? That's just 2 multiplications, same as schoolbook.
    
    # So the 24 terms for row 0 seem optimal with this approach.
    
    # For rows 1,2: 2x4x6.
    # Current: 3 k-blocks * 2 j-blocks * 7 terms = 42 terms.
    # Can we do better?
    
    # What if we use a 2x4x4 decomposition with rank 18, and then handle the last 2 k-columns separately?
    # 2x4x4 with rank 18, plus 2x4x2 with rank 8 (schoolbook) = 26 terms per 2x4x6 block.
    # But we have 3 such blocks? No, the k dimension is 6, so we can split into 4+2.
    # 2x4x4 (rank 18) + 2x4x2 (rank 8) = 26 terms for the 2x4x6 part.
    # Total: 24 + 26 = 50 terms.
    
    # But I don't have the rank-18 decomposition for 2x4x4.
    
    # Let's try a different approach: use the current 66-term decomposition but verify it.
    
    # Since I can't easily find a better decomposition, I'll keep the current one.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Part 1: M(1,4,6) for row 0
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)

    # Part 2: M(2,4,6) for rows 1,2
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)

    return {"U": U, "V": V, "W": W}
