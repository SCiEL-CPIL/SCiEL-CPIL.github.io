"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # Constructed by combining:
    # 1. Strassen (rank 7) for 2x2x2 part
    # 2. Schoolbook (rank 4) for 2x1x2 part (middle column)
    # 3. Schoolbook (rank 6) for 2x3x1 part (last column)
    
    # Total rank: 7 + 4 + 6 = 17
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3
    
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5
    
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify exact decomposition
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Build the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and check if we can remove one rank-1 term while maintaining exactness
    
    # For each pair of terms (r1, r2), check if their contributions can be merged
    # by finding coefficients alpha, beta such that the combined term works
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Check if any single term can be eliminated (its contribution is zero or redundant)
    for r_remove in range(rank):
        if time.time() > deadline - 5:
            break
        # Temporarily zero out term r_remove and see if tensor still matches
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, r_remove] = 0
        V_test[:, r_remove] = 0
        W_test[:, r_remove] = 0
        
        # Check if the removed term contributed nothing
        contributed = False
        for i in range(n * m):
            if abs(U[i, r_remove]) > 1e-12:
                contributed = True
                break
        if not contributed:
            continue
            
        # Full verification would be expensive; use a quick check
        # Check a subset of (i,j,k) tuples
        ok = True
        for i in range(2):
            for k in range(3):
                for j1 in range(3):
                    for j2 in range(3):
                        s = 0.0
                        for r in range(rank):
                            if r == r_remove:
                                continue
                            s += W[k*2+i, r] * U[i*3+j1, r] * V[j2*3+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Found a removable term
            new_rank = rank - 1
            idx = [r for r in range(rank) if r != r_remove]
            best_U = U[:, idx]
            best_V = V[:, idx]
            best_W = W[:, idx]
            best_rank = new_rank
            break
    
    # Try pairwise merging: can two rank-1 terms be replaced by one?
    # This is harder; try a limited search
    if best_rank == rank and time.time() < deadline - 10:
        # For the 6 schoolbook terms in Part 3 (indices 11-16), try to see
        # if any two can be combined. The structure is:
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        # These are independent dot products, so unlikely to merge.
        
        # Instead, try to see if we can modify the Strassen part to absorb
        # some of the schoolbook terms. This requires a more sophisticated search.
        pass
    
    # Final verification of best result
    if best_rank < rank:
        if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
            # Revert if verification fails
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
