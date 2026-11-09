"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110
    
    # We'll build a decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the rest.
    
    # The tensor is T[(i,j),(j',k),(i',k')] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # We partition the indices into blocks.
    
    # For 4x5x5:
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # Strategy: partition i into {0,1} and {2,3}
    # partition j into {0,1}, {2,3}, {4}
    # partition k into {0,1}, {2,3}, {4}
    
    # This gives us blocks:
    # For each i-block (size 2), j-block, k-block:
    #   If j-block size 2 and k-block size 2: use Strassen (rank 7)
    #   Otherwise: use schoolbook (rank = size_i * size_j * size_k)
    
    # Let's define the partitions:
    i_blocks = [(0, 2), (2, 4)]  # two blocks of size 2
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # two blocks of size 2, one of size 1
    k_blocks = [(0, 2), (2, 4), (4, 5)]  # two blocks of size 2, one of size 1
    
    # Build the decomposition
    U = {}  # (i,j) -> list of (coeff, rank_index)
    V = {}  # (j,k) -> list of (coeff, rank_index)
    W = {}  # (k,i) -> list of (coeff, rank_index)
    
    rank = 0
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict maps (i,j)->coeff, v_dict maps (j,k)->coeff, w_dict maps (k,i)->coeff"""
        nonlocal rank
        for (ij, cu) in u_dict.items():
            U.setdefault(ij, []).append((cu, rank))
        for (jk, cv) in v_dict.items():
            V.setdefault(jk, []).append((cv, rank))
        for (ki, cw) in w_dict.items():
            W.setdefault(ki, []).append((cw, rank))
        rank += 1
    
    def strassen_block(i0, j0, k0):
        """Add Strassen rank-7 decomposition for 2x2x2 block starting at (i0,j0,k0)."""
        # Strassen's 7 products for multiplying 2x2 by 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        
        # C00 = m1 - m4 + m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # m1 = (a00+a11)(b00+b11)
        add_term(
            {(i0,j0): 1, (i0+1,j0+1): 1},
            {(j0,k0): 1, (j0+1,k0+1): 1},
            {(k0,i0): 1, (k0+1,i0+1): 1}
        )
        # m2 = (a10+a11)*b00
        add_term(
            {(i0+1,j0): 1, (i0+1,j0+1): 1},
            {(j0,k0): 1},
            {(k0,i0+1): 1, (k0+1,i0+1): -1}
        )
        # m3 = a00*(b01-b11)
        add_term(
            {(i0,j0): 1},
            {(j0,k0+1): 1, (j0+1,k0+1): -1},
            {(k0+1,i0): 1, (k0+1,i0+1): 1}
        )
        # m4 = a11*(b10-b00)
        add_term(
            {(i0+1,j0+1): 1},
            {(j0+1,k0): 1, (j0,k0): -1},
            {(k0,i0+1): 1, (k0+1,i0+1): 1}
        )
        # m5 = (a00+a01)*b11
        add_term(
            {(i0,j0): 1, (i0,j0+1): 1},
            {(j0+1,k0+1): 1},
            {(k0+1,i0): -1, (k0+1,i0+1): 1}
        )
        # m6 = (a10-a00)*(b00+b01)
        add_term(
            {(i0+1,j0): 1, (i0,j0): -1},
            {(j0,k0): 1, (j0,k0+1): 1},
            {(k0+1,i0+1): 1}
        )
        # m7 = (a01-a11)*(b10+b11)
        add_term(
            {(i0,j0+1): 1, (i0+1,j0+1): -1},
            {(j0+1,k0): 1, (j0+1,k0+1): 1},
            {(k0,i0): 1}
        )
    
    def schoolbook_block(i0, i1, j0, j1, k0, k1):
        """Add schoolbook rank-(i1-i0)*(j1-j0)*(k1-k0) for block."""
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    add_term(
                        {(i,j): 1},
                        {(j,k): 1},
                        {(k,i): 1}
                    )
    
    # Process all block combinations
    for (i0, i1) in i_blocks:
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                if (i1 - i0) == 2 and (j1 - j0) == 2 and (k1 - k0) == 2:
                    strassen_block(i0, j0, k0)
                else:
                    schoolbook_block(i0, i1, j0, j1, k0, k1)
    
    # Now build the output matrices
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), terms in U.items():
        for (coeff, r) in terms:
            U_mat[i * m + j][r] += coeff
    
    for (j, k), terms in V.items():
        for (coeff, r) in terms:
            V_mat[j * p + k][r] += coeff
    
    for (k, i), terms in W.items():
        for (coeff, r) in terms:
            W_mat[k * n + i][r] += coeff
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U_mat[i*m+j][c] * V_mat[jp*p+k][c] * W_mat[kp*n+ip][c]
                                target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                err = abs(s - target)
                                if err > max_err:
                                    max_err = err
        return max_err
    
    err = verify(U_mat, V_mat, W_mat, rank)
    
    if err > 1e-9:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
