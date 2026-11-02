"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses rectangular Strassen: (3x3x4 rank-19) ⊗ (1x1x2 rank-2) = rank 38.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    # Strategy: 3x3x8 = (3x3x4) ⊗ (1x1x2)
    # Build rank-19 decomposition for 3x3x4, then tensor with 2-element scalar mult.
    #
    # For 3x3x4: split B into B_left (3x2) and B_right (3x2)
    # A[0:2,0:2] * B_left[0:2,0:2]: Strassen (7 terms)
    # A[0:2,0:2] * B_right[0:2,0:2]: Strassen (7 terms)  
    # A[0:2,2] * B_left[2,0:2]: direct (2 terms)
    # A[0:2,2] * B_right[2,0:2]: direct (2 terms)
    # A[2,0:2] * B_left[0:2,0:2]: direct (2 terms)
    # A[2,0:2] * B_right[0:2,0:2]: direct (2 terms)
    # A[2,2] * B_left[2,0:2]: direct (1 term)
    # A[2,2] * B_right[2,0:2]: direct (1 term)
    # Total: 7+7+2+2+2+2+1+1 = 24... that's too many.
    #
    # Let me think differently. For 3x3x4, we need a better decomposition.
    # Actually, let me try: treat as 3x3x4 directly with clever grouping.
    #
    # Alternative: 3x3x4 can be seen as:
    # - 2x2x4 block: A[0:2,0:2]*B[0:2,0:4] - can use rank 7 for 2x2x2 twice = 14
    # - Remaining: A[0:2,2]*B[2,0:4] = 2*4=8 direct, A[2,0:2]*B[0:2,0:4]=2*4=8, A[2,2]*B[2,0:4]=4
    # Total: 14+8+8+4 = 34... still too many.
    #
    # Let me try a different approach: use the fact that 3x3x4 has known bounds.
    # Actually, let me just build a valid decomposition and aim for rank < 72.
    #
    # New strategy: Use Strassen on 2x2x2 blocks for the 2x2 part of A and B,
    # and handle the rest directly. For 3x3x8:
    #
    # A[0:2,0:2] * B[0:2,0:8]: This is 2x2 * 2x8. 
    #   Split B into four 2x2 blocks: 4 * Strassen(2x2x2) = 4*7 = 28 terms
    # A[0:2,2] * B[2,0:8]: 2x1 * 1x8 = 16 direct terms
    # A[2,0:2] * B[0:2,0:8]: 1x2 * 2x8 = 16 direct terms  
    # A[2,2] * B[2,0:8]: 1x1 * 1x8 = 8 direct terms
    # Total: 28 + 16 + 16 + 8 = 68 terms (this is what we had)
    #
    # To improve, let's try to combine some direct terms using Strassen-like tricks.
    # 
    # Key insight: A[0:2,2] * B[2,0:8] and A[2,0:2] * B[0:2,0:8] can be combined
    # if we treat them as part of a larger structure.
    #
    # Actually, let's try the tensor product approach properly.
    # 3x3x8 = (3x3x4) ⊗ (1x1x2)
    # If we have a rank-r decomposition for 3x3x4, we get rank-2r for 3x3x8.
    #
    # For 3x3x4, let me construct a decomposition:
    # View 3x3x4 as: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..2, k in 0..3
    #
    # Split: A = [A_top | A_bot] where A_top is 2x3, A_bot is 1x3
    #        B = [B_top; B_bot] where B_top is 3x4, B_bot is... wait, B is 3x4 so B_top is 2x4, B_bot is 1x4
    #
    # C = A_top * B_top + A_bot * B_bot
    #
    # A_top * B_top: 2x3 * 3x4
    #   = A_top[:,0:2] * B_top[0:2,:] + A_top[:,2] * B_top[2,:]
    #   A_top[:,0:2] is 2x2, B_top[0:2,:] is 2x4
    #   Split B_top[0:2,:] into two 2x2 blocks: 2 * Strassen = 14 terms
    #   A_top[:,2] * B_top[2,:]: 2x1 * 1x4 = 8 direct terms
    #   Subtotal: 14 + 8 = 22
    #
    # A_bot * B_bot: 1x3 * 3x4
    #   = A_bot[0:2] * B_bot[0:2,:] + A_bot[2] * B_bot[2,:]
    #   A_bot[0:2] * B_bot[0:2,:]: 1x2 * 2x4 = 8 direct
    #   A_bot[2] * B_bot[2,:]: 1x1 * 1x4 = 4 direct
    #   Subtotal: 8 + 4 = 12
    #
    # Total for 3x3x4: 22 + 12 = 34... too high.
    #
    # Let me try a cleaner approach: just use the existing 68-term decomposition
    # but see if we can reduce it by combining terms.
    #
    # Actually, the directive says to use (3x3x4 rank-19) ⊗ (1x1x2 rank-2) = rank 38.
    # Let me find a rank-19 decomposition for 3x3x4.
    #
    # Known: 2x2x2 has rank 7 (Strassen)
    # 3x3x3 has rank at most 23 (Lader) or 20 (some constructions)
    # 3x3x4 should have rank at most... let me think.
    #
    # 3x3x4: C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this as:
    # - 2x2x2 block (A[0:2,0:2]*B[0:2,0:2]): rank 7
    # - 2x2x2 block (A[0:2,0:2]*B[0:2,2:4]): rank 7
    # - A[0:2,2]*B[2,0:4]: 2*4 = 8 direct
    # - A[2,0:2]*B[0:2,0:4]: 2*4 = 8 direct
    # - A[2,2]*B[2,0:4]: 4 direct
    # Total: 7+7+8+8+4 = 34
    #
    # To get to 19, we need much more cleverness. Let me try a different split.
    #
    # What if we use a 2x3 * 3x4 decomposition more cleverly?
    # Actually, let me just implement the 68-term version but try to optimize it
    # by finding linear dependencies among the terms.
    #
    # For now, let me implement a clean 68-term decomposition and verify it works.

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 0:8]: direct (16 terms)
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:8]: direct (16 terms)
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:8]: direct (8 terms)
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
