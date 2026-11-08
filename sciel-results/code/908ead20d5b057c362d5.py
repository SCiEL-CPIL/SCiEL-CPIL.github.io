import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition patterns (indexed by local i,j,k in {0,1})
    # U[i,j], V[j,k], W[k,i]
    strassen_U = [
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 0, 1), (1, 1, -1)]
    ]
    strassen_V = [
        [(0, 0, 1), (0, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 1, 1)]
    ]
    strassen_W = [
        [(0, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(0, 0, -1), (1, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    def verify(U_arr, V_arr, W_arr, rank):
        """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def build_rank56():
        """Standard 8 independent Strassen blocks, rank 56."""
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        block_idx = 0
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                base = block_idx * 7
                for r in range(7):
                    for (li, lj, val) in strassen_U[r]:
                        gi = li + 0  # i_offset always 0 since n=2
                        gj = lj + j_off
                        U[gi * m + gj, base + r] += val
                    for (lj, lk, val) in strassen_V[r]:
                        gj = lj + j_off
                        gk = lk + k_off
                        V[gj * p + gk, base + r] += val
                    for (lk, li, val) in strassen_W[r]:
                        gk = lk + k_off
                        gi = li + 0
                        W[gk * n + gi, base + r] += val
                block_idx += 1
        return U, V, W, rank
    
    def build_rank48():
        """
        Rank-48 scheme: use two 2x4x4 rank-24 decompositions.
        For 2x4x4: split j into {0,1} and {2,3}, k into {0,1} and {2,3}.
        Four 2x2x2 sub-problems. Share 4 Strassen terms between the two k-blocks
        that have the same j-block.
        
        Strategy: For j-block 0 (j=0,1), we have two k-blocks (k=0,1 and k=2,3).
        Use 7+7=14 terms but share 7 terms? No, that gives 14.
        
        Better: Use a rank-12 scheme for 2x4x2? No.
        
        Alternative: Use the fact that for 2x4x4, we can use a scheme with rank 24
        by sharing terms between the four 2x2x2 sub-problems.
        
        Let's use: 4 sub-problems each with 7 terms = 28, then merge 4 pairs = 24.
        """
        rank = 48
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # For each half of k (k=0..3 and k=4..7), build a 2x4x4 rank-24 block
        for k_half in range(2):
            k_base = k_half * 4
            block_start = k_half * 24
            
            # 4 sub-problems: (j_off, k_off_local) where j_off in {0,2}, k_off_local in {0,2}
            # Assign rank slots:
            # Sub-problem (0,0): slots 0-6
            # Sub-problem (0,1): slots 7-13  
            # Sub-problem (1,0): slots 14-20
            # Sub-problem (1,1): slots 21-27
            # Then merge 4 pairs to get rank 24
            
            # First, build all 4 independent Strassen blocks (rank 28)
            U_temp = np.zeros((n * m, 28))
            V_temp = np.zeros((m * p, 28))
            W_temp = np.zeros((p * n, 28))
            
            sp_idx = 0
            for j_off in [0, 2]:
                for k_off_local in [0, 2]:
                    k_off_global = k_base + k_off_local
                    base = sp_idx * 7
                    for r in range(7):
                        for (li, lj, val) in strassen_U[r]:
                            gi = li
                            gj = lj + j_off
                            U_temp[gi * m + gj, base + r] += val
                        for (lj, lk, val) in strassen_V[r]:
                            gj = lj + j_off
                            gk = lk + k_off_global
                            V_temp[gj * p + gk, base + r] += val
                        for (lk, li, val) in strassen_W[r]:
                            gk = lk + k_off_global
                            gi = li
                            W_temp[gk * n + gi, base + r] += val
                    sp_idx += 1
            
            # Now try to merge 4 pairs of terms to reduce from 28 to 24
            # Merge strategy: for terms with same U and V but different W, or vice versa
            # Simple approach: check if any two terms can be combined
            # For now, just use rank 28 per half = 56 total (no improvement)
            # 
            # Alternative: use a different scheme entirely
            # 
            # Let's just copy the 28 terms for now (no merging)
            U[block_start:block_start+28] = U_temp
            V[block_start:block_start+28] = V_temp
            W[block_start:block_start+28] = W_temp
            rank = 56  # No improvement, still 56
        
        return U, V, W, rank
    
    # Build the rank-56 baseline
    U_arr, V_arr, W_arr, rank = build_rank56()
    
    # Verify
    max_err = verify(U_arr, V_arr, W_arr, rank)
    
    if max_err > 1e-10:
        # Fallback: trivial rank-64
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i*m+j, idx] = 1
                    V_t[j*p+k, idx] = 1
                    W_t[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
