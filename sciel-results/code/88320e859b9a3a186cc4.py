"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Flatten T to get the vector of non-zero entries
    # We need to find rank-1 tensors that sum to T
    # Let's try to find a decomposition with rank R < 72
    
    # Strategy: Try to use Strassen-style block decomposition
    # For [3,3,8], we can split p into 4+4
    # Each [3,3,4] can be decomposed further
    
    # Let's try a direct approach: formulate as integer linear programming
    # but that's too complex. Instead, let's use known results or try specific constructions.
    
    # Known lower bounds for [3,3,8]:
    # The tensor rank of [3,3,8] is at least 33 (by Strassen's bound or similar)
    # But let's try to find a good upper bound.
    
    # Try a block decomposition:
    # Split p=8 into two blocks of 4
    # For each [3,3,4] block, try to use a decomposition
    
    # Let's try a simple improvement over schoolbook:
    # Use the fact that [2,2,2] has rank 7 (Strassen)
    
    # Split [3,3,8] into smaller pieces:
    # We can write [3,3,8] = [1,3,8] + [2,3,8]
    # [1,3,8]: rank 24 (schoolbook, since one dimension is 1)
    # [2,3,8]: split into [2,1,8] + [2,2,8]
    #   [2,1,8]: rank 16
    #   [2,2,8]: split into [2,2,4] + [2,2,4]
    #     [2,2,4]: split into [2,2,2] + [2,2,2]
    #       [2,2,2]: rank 7
    #       So [2,2,4]: rank 14
    #     So [2,2,8]: rank 28
    #   So [2,3,8]: rank 16 + 28 = 44
    # Total: 24 + 44 = 68
    
    # Let's implement this decomposition.
    
    # First, let's define a helper to create the schoolbook decomposition for [n,m,p]
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # For [2,2,2], use Strassen's decomposition
    def strassen_222():
        # Strassen's algorithm for 2x2 matrix multiplication
        # U is 4x7, V is 4x7, W is 4x7
        # The 7 rank-1 terms
        U = np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0]
        ])
        V = np.array([
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        W = np.array([
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0]
        ])
        # Wait, I need to verify this is correct. Let me use the standard Strassen formulas.
        # For C = A*B where A, B, C are 2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+cg, bf+dh], [ce+dg, af+bh]]
        # Strassen's 7 products:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+h)
        # m7 = (b-d)(e+h)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # Let me index: A = A[0,0], A[0,1], A[1,0], A[1,1]
        # B = B[0,0], B[0,1], B[1,0], B[1,1]
        # C = C[0,0], C[0,1], C[1,0], C[1,1]
        
        # U rows correspond to A entries: [a, b, c, d]
        # V rows correspond to B entries: [e, f, g, h]
        # W rows correspond to C entries: [C11, C12, C21, C22]
        
        # m1 = (a+d)(e+h): U contribution to m1: a gets 1, d gets 1
        #                            V contribution to m1: e gets 1, h gets 1
        # m2 = (c+d)e: U: c gets 1, d gets 1
        #              V: e gets 1
        # m3 = a(f-h): U: a gets 1
        #             V: f gets 1, h gets -1
        # m4 = d(g-e): U: d gets 1
        #             V: g gets 1, e gets -1
        # m5 = (a+b)h: U: a gets 1, b gets 1
        #              V: h gets 1
        # m6 = (c-d)(f+h): U: c gets 1, d gets -1
        #                 V: f gets 1, h gets 1
        # m7 = (b-d)(e+h): U: b gets 1, d gets -1
        #                 V: e gets 1, h gets 1
        
        U = np.array([
            # a, b, c, d
            [1, 0, 0, 0],  # a
            [0, 1, 0, 0],  # b
            [0, 0, 1, 0],  # c
            [0, 0, 0, 1]   # d
        ])
        
        V = np.array([
            # e, f, g, h
            [1, 0, 0, 0],  # e
            [0, 1, 0, 0],  # f
            [0, 0, 1, 0],  # g
            [0, 0, 0, 1]   # h
        ])
        
        # W: for each of the 7 products, what is the contribution to C11, C12, C21, C22
        # m1: C11 += 1, C22 += 1
        # m2: C21 += 1, C22 -= 1
        # m3: C12 += 1, C22 += 1
        # m4: C11 += 1, C21 += 1
        # m5: C11 -= 1, C12 += 1
        # m6: C22 += 1
        # m7: C11 += 1
        W = np.array([
            # m1, m2, m3, m4, m5, m6, m7
            [1, 0, 0, 1, -1, 0, 1],  # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 0, 0, 1, 0]   # C22
        ])
        
        return U, V, W
    
    # Now let's build the full decomposition for [3,3,8]
    # We'll accumulate U, V, W matrices
    
    # Start with empty lists
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_block, m_block, p_block, i_offset, j_offset, k_offset):
        """Add a schoolbook decomposition for a sub-block."""
        U_sub, V_sub, W_sub = schoolbook(n_block, m_block, p_block)
        U_list.append(U_sub)
        V_list.append(V_sub)
        W_list.append(W_sub)
    
    # Helper to add Strassen decomposition for [2,2,2]
    def add_strassen(i_offset, j_offset, k_offset):
        """Add Strassen decomposition for a [2,2,2] block."""
        U_sub, V_sub, W_sub = strassen_222()
        U_list.append(U_sub)
        V_list.append(V_sub)
        W_list.append(W_sub)
    
    # Let's use the block decomposition:
    # [3,3,8] = [1,3,8] + [2,3,8]
    # [1,3,8]: schoolbook, rank 24
    # [2,3,8] = [2,1,8] + [2,2,8]
    #   [2,1,8]: schoolbook, rank 16
    #   [2,2,8] = [2,2,4] + [2,2,4]
    #     [2,2,4] = [2,2,2] + [2,2,2]
    #       [2,2,2]: Strassen, rank 7 each, so [2,2,4] has rank 14
    
    # For [1,3,8]: i in [0], j in [0,1,2], k in [0..7]
    add_block(1, 3, 8, 0, 0, 0)
    
    # For [2,1,8]: i in [1,2], j in [0], k in [0..7]
    add_block(2, 1, 8, 1, 0, 0)
    
    # For [2,2,8]: i in [1,2], j in [1,2], k in [0..7]
    # Split into two [2,2,4] blocks
    # First [2,2,4]: k in [0..3]
    #   Split into two [2,2,2]
    #   First [2,2,2]: k in [0,1]
    add_strassen(1, 1, 0)
    # Second [2,2,2]: k in [2,3]
    add_strassen(1, 1, 2)
    
    # Second [2,2,4]: k in [4..7]
    #   First [2,2,2]: k in [4,5]
    add_strassen(1, 1, 4)
    #   Second [2,2,2]: k in [6,7]
    add_strassen(1, 1, 6)
    
    # Now concatenate all the U, V, W matrices
    U_full = np.hstack(U_list)
    V_full = np.hstack(V_list)
    W_full = np.hstack(W_list)
    
    # The dimensions should be:
    # U_full: (9, rank)
    # V_full: (24, rank)
    # W_full: (24, rank)
    
    # But wait, the blocks are not aligned correctly. The offsets need to be applied.
    # Let me redo this more carefully.
    
    # Actually, the approach above is flawed because I'm not properly offsetting the indices.
    # Let me try a different approach: just use the schoolbook decomposition for now,
    # but try to find a better one through optimization.
    
    # For now, return the schoolbook decomposition as a baseline
    # and we can improve from there.
    
    # Let's just return the schoolbook for now
    U, V, W = schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
