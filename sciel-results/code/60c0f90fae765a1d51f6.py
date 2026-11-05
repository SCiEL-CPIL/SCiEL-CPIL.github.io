"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # The tensor T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    # We want U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Use block decomposition with Strassen for 2x2x2 blocks
    # and schoolbook for the rest. Then try to merge columns where possible.
    
    def verify(U, V, W, rank):
        """Verify the decomposition is correct."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col):
        """Add schoolbook terms for a block."""
        for di in range(hi):
            for dj in range(hj):
                for dk in range(hk):
                    U[(i0+di)*m + (j0+dj)][col] += 1
                    V[(j0+dj)*p + (k0+dk)][col] += 1
                    W[(k0+dk)*n + (i0+di)][col] += 1
                    col += 1
        return col
    
    def strassen_block(i0, j0, k0, U, V, W, col):
        """Add Strassen terms for a 2x2x2 block."""
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each term t has: U coefficients, V coefficients, W coefficients
        # We encode each as a list of (row, coeff) for U, V, W
        
        strassen_terms = [
            # (U_rows, V_rows, W_rows)
            # M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
            ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
            # M2 = (A10+A11)(B00) -> contributes to C10(+1), C11(-1)
            ([((1,0),1), ((1,1),1)], [((0,0),1)], [((0,1),1), ((1,1),-1)]),
            # M3 = (A00)(B10-B11) -> contributes to C01(+1), C11(+1)
            ([((0,0),1)], [((1,0),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
            # M4 = (A11)(B01-B00) -> contributes to C00(+1), C10(+1)
            ([((1,1),1)], [((0,1),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
            # M5 = (A00+A01)(B11) -> contributes to C00(-1), C01(+1)
            ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
            # M6 = (A10-A00)(B00+B01) -> contributes to C11(+1)
            ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
            # M7 = (A01-A11)(B10+B11) -> contributes to C00(+1)
            ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
        ]
        
        for u_terms, v_terms, w_terms in strassen_terms:
            for (di, dj), coeff in u_terms:
                U[(i0+di)*m + (j0+dj)][col] += coeff
            for (dj, dk), coeff in v_terms:
                V[(j0+dj)*p + (k0+dk)][col] += coeff
            for (dk, di), coeff in w_terms:
                W[(k0+dk)*n + (i0+di)][col] += coeff
            col += 1
        
        return col
    
    # Try different block decompositions
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate candidate splits
    # n=3: possible splits: [3], [1,2], [2,1], [1,1,1]
    # m=5: possible splits: [5], [1,4], [4,1], [2,3], [3,2], [1,1,3], [1,3,1], [3,1,1], [2,2,1], [2,1,2], [1,2,2], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
    # p=7: similar
    
    def get_splits(total, max_parts=4):
        """Get all compositions of 'total' with at most max_parts parts."""
        if total == 0:
            return [[]]
        if total == 1:
            return [[1]]
        if max_parts == 1:
            return [[total]]
        
        result = []
        for first in range(1, total + 1):
            for rest in get_splits(total - first, max_parts - 1):
                result.append([first] + rest)
        return result
    
    # For efficiency, limit the number of splits we try
    splits_n = [[3], [1,2], [2,1], [1,1,1]]
    splits_m = [[5], [1,4], [4,1], [2,3], [3,2], [2,2,1], [1,2,2], [2,1,2], [1,1,3], [3,1,1], [1,3,1]]
    splits_p = [[7], [1,6], [6,1], [2,5], [5,2], [3,4], [4,3], [2,2,3], [3,2,2], [2,3,2], [1,3,3], [3,3,1], [1,1,5], [5,1,1], [1,5,1], [2,2,2,1]]
    
    # Limit total combinations to avoid timeout
    max_combos = 200
    combo_count = 0
    
    for sn in splits_n:
        if time.time() > deadline - 10:
            break
        for sm in splits_m:
            if time.time() > deadline - 10:
                break
            for sp in splits_p:
                if combo_count >= max_combos:
                    break
                if time.time() > deadline - 10:
                    break
                combo_count += 1
                
                # Calculate expected rank
                rank_estimate = 0
                i_offsets = [0]
                for s in sn[:-1]:
                    i_offsets.append(i_offsets[-1] + s)
                j_offsets = [0]
                for s in sm[:-1]:
                    j_offsets.append(j_offsets[-1] + s)
                k_offsets = [0]
                for s in sp[:-1]:
                    k_offsets.append(k_offsets[-1] + s)
                
                for ii in range(len(sn)):
                    for jj in range(len(sm)):
                        for kk in range(len(sp)):
                            hi = sn[ii]
                            hj = sm[jj]
                            hk = sp[kk]
                            if hi == 2 and hj == 2 and hk == 2:
                                rank_estimate += 7  # Strassen
                            else:
                                rank_estimate += hi * hj * hk  # schoolbook
                
                if rank_estimate >= best_rank:
                    continue
                
                # Build the decomposition
                U = [[0] * rank_estimate for _ in range(n * m)]
                V = [[0] * rank_estimate for _ in range(m * p)]
                W = [[0] * rank_estimate for _ in range(p * n)]
                
                col = 0
                for ii in range(len(sn)):
                    for jj in range(len(sm)):
                        for kk in range(len(sp)):
                            hi = sn[ii]
                            hj = sm[jj]
                            hk = sp[kk]
                            i0 = i_offsets[ii]
                            j0 = j_offsets[jj]
                            k0 = k_offsets[kk]
                            
                            if hi == 2 and hj == 2 and hk == 2:
                                col = strassen_block(i0, j0, k0, U, V, W, col)
                            else:
                                col = schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col)
                
                # Verify
                if col == rank_estimate:
                    if verify(U, V, W, rank_estimate):
                        best_rank = rank_estimate
                        best_result = {"U": U, "V": V, "W": W}
    
    # If no improvement found, use schoolbook
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
