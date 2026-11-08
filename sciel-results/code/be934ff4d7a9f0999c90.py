import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1 (j=0,1): 3 k-pairs * 7 (Strassen) = 21
    # Part 2 (j=2,3,4): 3 k-pairs * (7 (Strassen for j=2,3) + 1 (Rank-1 for j=4)) = 3 * 8 = 24
    # Total Rank = 45
    
    R = 45
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    rank_counter = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a Strassen decomposition for a 2x2x2 block 
        located at global indices starting at i0, j0, k0.
        Returns the number of rank components used (7).
        """
        global rank_counter
        if rank_counter + 7 > R:
            return 0
            
        # Strassen intermediate products for local indices 0,1
        # M1 = A00 * (B01 + B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 - A11) * (B00 + B01)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * (B00 - B10)
        # M6 = (A10 - A00) * (B10 + B11)
        # M7 = (A01 + A11) * (B01 - B11)
        
        # Output reconstruction:
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M2
        # C10 = M3 + M4
        # C11 = M1 - M5 + M3 + M7
        
        # We define 7 rank-1 tensors.
        # For each, we specify which U, V, W indices are nonzero and with what value.
        
        strassen_defs = [
            # (U_indices, V_indices, W_indices, Coeffs)
            # M1
            ([(i0+0, j0+0)], 
             [(j0+0, k0+1), (j0+1, k0+1)], 
             [(k0+1, i0+0), (k0+1, i0+1)], 
             [1, 1, 1, 1]),
            # M2
            ([(i0+0, j0+0), (i0+1, j0+0)], 
             [(j0+1, k0+1)], 
             [(k0+1, i0+0), (k0+1, i0+1)], 
             [1, 1, 1, 1]),
            # M3
            ([(i0+0, j0+1), (i0+1, j0+1)], 
             [(j0+0, k0+0), (j0+0, k0+1)], 
             [(k0+0, i0+1), (k0+1, i0+1)], 
             [1, 1, 1, 1]), # Note: C10 and C11 get +1 from M3
            # M4
            ([(i0+1, j0+1)], 
             [(j0+1, k0+0), (j0+0, k0+0)], 
             [(k0+0, i0+0), (k0+0, i0+1)], 
             [1, 1, 1, 1]), # C00 and C10 get +1 from M4
            # M5
            ([(i0+0, j0+0), (i0+0, j0+1)], 
             [(j0+0, k0+0), (j0+1, k0+0)], 
             [(k0+0, i0+0), (k0+1, i0+1)], 
             [1, 1, 1, -1]), # C00 gets +1, C11 gets -1
            # M6
            ([(i0+1, j0+0), (i0+0, j0+0)], 
             [(j0+1, k0+0), (j0+1, k0+1)], 
             [(k0+0, i0+0)], 
             [1, 1, 1]), # C00 gets +1
            # M7
            ([(i0+0, j0+1), (i0+1, j0+1)], 
             [(j0+0, k0+1), (j0+1, k0+1)], 
             [(k0+1, i0+1)], 
             [1, 1, 1]), # C11 gets +1
        ]
        
        for u_idxs, v_idxs, w_idxs, coeffs in strassen_defs:
            if rank_counter >= R:
                break
            r = rank_counter
            # Apply U
            for idx in u_idxs:
                U[idx[0] * m + idx[1], r] += coeffs[0] # Wait, coeffs list length must match total non-zeros?
                # Actually, my definition above groups them. 
                # It is safer to iterate through each non-zero entry individually.
            
            # Let's rewrite the addition logic to be robust.
            # The list `coeffs` in my quick sketch was ambiguous. 
            # Let's define each rank-1 component explicitly.
            
            rank_counter += 1
            
        # The logic above is flawed. Let's do it properly.
        # I will rollback the rank_counter and do it correctly.
        rank_counter -= 7 
        
        explicit_strassen = [
            # M1: A00 (B01+B11) -> C01(+1), C11(+1)
            ({(0,0): 1}, {(0,1): 1, (1,1): 1}, {(1,0): 1, (1,1): 1}),
            # M2: (A00+A10) B11 -> C01(+1), C11(+1)
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
            # M3: (A01-A11) (B00+B01) -> C10(+1), C11(+1)
            ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1, (1,1): 1}),
            # M4: A11 (B10-B00) -> C00(+1), C10(+1)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5: (A00+A01) (B00-B10) -> C00(+1), C11(-1)
            ({(0,0): 1, (0,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1, (1,1): -1}),
            # M6: (A10-A00) (B10+B11) -> C00(+1)
            ({(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            # M7: (A01+A11) (B01-B11) -> C11(+1)
            ({(0,1): 1, (1,1): 1}, {(0,1): 1, (1,1): -1}, {(1,1): 1}),
        ]
        
        for u_dict, v_dict, w_dict in explicit_strassen:
            r = rank_counter
            for (li, lj), val in u_dict.items():
                U[(i0 + li) * m + (j0 + lj), r] += val
            for (lj, lk), val in v_dict.items():
                V[(j0 + lj) * p + (k0 + lk), r] += val
            for (lk, li), val in w_dict.items():
                W[(k0 + lk) * n + (i0 + li), r] += val
            rank_counter += 1
            
        return 7

    def add_rank1_block(i0, j0, k0):
        """
        Adds a rank-1 decomposition for a 2x1x2 block 
        located at global indices starting at i0, j0, k0.
        The '1' dimension is j.
        Returns 1.
        """
        global rank_counter
        if rank_counter >= R:
            return 0
        
        r = rank_counter
        # A is 2x1, B is 1x2, C is 2x2
        # C_i,k = A_i * B_k
        # Tensor T[i, j, k] = 1 for all i,k, j=0.
        # U[i, 0] = 1, V[0, 0] = 1, W[k, i] = 1?
        # Sum U[i] V[0] W[k,i] = 1*1*1 = 1.
        # U indices: (i0+0, j0), (i0+1, j0)
        # V indices: (j0, k0+0), (j0, k0+1) -- Wait, V is (m, p). j is row, k is col.
        # V[j0, k0+0] = 1, V[j0, k0+1] = 1
        # W indices: W is (p, n). k is row, i is col.
        # W[k0+0, i0+0] = 1, W[k0+0, i0+1] = 1
        # W[k0+1, i0+0] = 1, W[k0+1, i0+1] = 1
        
        U[(i0+0) * m + j0, r] += 1
        U[(i0+1) * m + j0, r] += 1
        
        V[j0 * p + (k0+0), r] += 1
        V[j0 * p + (k0+1), r] += 1
        
        W[(k0+0) * n + (i0+0), r] += 1
        W[(k0+0) * n + (i0+1), r] += 1
        W[(k0+1) * n + (i0+0), r] += 1
        W[(k0+1) * n + (i0+1), r] += 1
        
        rank_counter += 1
        return 1

    # Part 1: j in {0, 1}
    # k pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        add_strassen_block(0, 0, k_start)
        
    # Part 2: j in {2, 3, 4}
    # k pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        # Sub-problem for j in {2, 3}: 2x2x2 (Strassen)
        # Local j=0 maps to global j=2, Local j=1 maps to global j=3
        add_strassen_block(0, 2, k_start)
        
        # Sub-problem for j in {4}: 2x1x2 (Rank 1)
        add_rank1_block(0, 4, k_start)

    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
