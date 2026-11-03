"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # First, build the rank-17 decomposition
    rank17 = 17
    U17 = np.zeros((n * m, rank17))
    V17 = np.zeros((m * p, rank17))
    W17 = np.zeros((p * n, rank17))
    
    # Part 1: Strassen for 2x2x2
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for middle column
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    # Part 3: Schoolbook for last column
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
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
    
    # Verify rank-17 is correct
    assert _verify(U17, V17, W17, n, m, p, rank17), "Rank-17 decomposition failed verification"
    
    # Try to find a rank-16 decomposition by searching for linear dependencies
    # Strategy: try to merge two rank-17 factors into one by finding
    # coefficients alpha, beta such that the combined factor still decomposes the tensor
    
    # Build the tensor T[i,j,k] = 1 if j==j' else 0 for all (i,j,j',k)
    # The tensor is: T[i, j, j', k] = delta_{j,j'}
    # We need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    
    # For rank 16, we need to find if there's a linear combination that works.
    # Try all pairs of columns to see if one can be expressed as a linear combination
    # of the others while maintaining the decomposition.
    
    # Alternative approach: use the structure to find a direct rank-16 decomposition
    # by exploiting that some products can be shared.
    
    # Let's try a different decomposition strategy:
    # Use 2x3x3 = (2x2x2) + (2x1x2) + (2x3x1) but look for savings
    
    # Actually, let's try to directly construct rank 16 by modifying the Strassen part
    # or by finding a smarter combination.
    
    # Try: Can we save one rank by combining the last column schoolbook with Strassen?
    
    # New approach: Try to find if rank 16 is possible by solving the system
    # Set up the equations for rank 16 and check if a solution exists with small integers
    
    # For efficiency, try specific structured rank-16 decompositions
    
    # Attempt 1: Modify Strassen to incorporate one column
    # Attempt 2: Use a different factorization pattern
    
    # Let's try a systematic search for rank 16 by perturbing the rank-17 solution
    # and checking if we can drop one column while maintaining exact decomposition
    
    best_rank = rank17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Try dropping each column and see if the remaining 16 can still decompose
    # This won't work directly since we need exact decomposition, but let's check
    # if any column is redundant (all zeros in the tensor contribution)
    
    # More promising: try to find if two columns can be merged
    # For each pair (a, b), check if column b can be written as alpha * column a
    # in all three matrices simultaneously
    
    for a in range(rank17):
        for b in range(a + 1, rank17):
            # Check if V[:, b] = c * V[:, a] for some c
            # and U[:, b] = c * U[:, a] and W[:, b] = c * W[:, a]
            # This is very restrictive and unlikely, but let's check
            
            # Actually, we need: the tensor contribution of column b equals
            # alpha * tensor contribution of column a
            # T_b[i,j,j',k] = U[i*m+j, b] * V[j'*p+k, b] * W[k*n+i, b]
            # We want T_b = alpha * T_a for all (i,j,j',k)
            
            # This means U[:, b] = alpha * U[:, a], V[:, b] = alpha * V[:, a], W[:, b] = alpha * W[:, a]
            # which is very restrictive.
            
            # More generally, we can have a linear combination:
            # T_new = T_a + beta * T_b, and drop both a and b, adding one new column
            # This is complex. Let's try a different approach.
            
            pass
    
    # Try a different strategy: construct rank 16 directly using known results
    # For 2x3x3, the minimal rank is known to be 16 (Bini et al.)
    
    # Let's try to construct it using the following pattern:
    # Split into overlapping 2x2x2 blocks with shared factors
    
    # Direct rank-16 construction attempt:
    # Use 16 factors where some factors serve multiple purposes
    
    rank16 = 16
    U16 = np.zeros((n * m, rank16))
    V16 = np.zeros((m * p, rank16))
    W16 = np.zeros((p * n, rank16))
    
    # Factor 1: (A00+A11)(B00+B11) -> C00, C11
    U16[0, 0] = 1; U16[4, 0] = 1
    V16[0, 0] = 1; V16[4, 0] = 1
    W16[0, 0] = 1; W16[3, 0] = 1
    
    # Factor 2: (A10+A11)B00 -> C10, -C11
    U16[3, 1] = 1; U16[4, 1] = 1
    V16[0, 1] = 1
    W16[1, 1] = 1; W16[3, 1] = -1
    
    # Factor 3: A00(B01-B11) -> C01, C11
    U16[0, 2] = 1
    V16[1, 2] = 1; V16[4, 2] = -1
    W16[2, 2] = 1; W16[3, 2] = 1
    
    # Factor 4: A11(B10-B00) -> C00, C10
    U16[4, 3] = 1
    V16[3, 3] = 1; V16[0, 3] = -1
    W16[0, 3] = 1; W16[1, 3] = 1
    
    # Factor 5: (A00+A01)B11 -> -C00, C01
    U16[0, 4] = 1; U16[1, 4] = 1
    V16[4, 4] = 1
    W16[0, 4] = -1; W16[2, 4] = 1
    
    # Factor 6: (A10-A00)(B00+B01) -> C11
    U16[3, 5] = 1; U16[0, 5] = -1
    V16[0, 5] = 1; V16[1, 5] = 1
    W16[3, 5] = 1
    
    # Factor 7: (A01-A11)(B10+B11) -> C00
    U16[1, 6] = 1; U16[4, 6] = -1
    V16[3, 6] = 1; V16[4, 6] = 1
    W16[0, 6] = 1
    
    # Now for the remaining 9 factors, we need to cover:
    # - A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (4 terms)
    # - A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2] (6 terms)
    # Total: 10 terms, but we only have 9 factors left
    # So we need to merge one pair.
    
    # Look for a merge: Can we combine A02*B20 and A00*B02 into one factor?
    # No, they have different indices.
    
    # Try: combine A02*B22 and A12*B22? No, different A indices.
    
    # Let's try a different grouping. What if we use a different Strassen variant?
    
    # Alternative: use the fact that we only need 2 rows of C, so we have more freedom.
    
    # Let's try to verify if the above rank-16 attempt works by completing it
    # and checking. If it doesn't work, fall back to rank 17.
    
    # Complete the rank-16 attempt with 9 more factors:
    # Factor 8: A02 B20 -> C00
    U16[2, 7] = 1
    V16[6, 7] = 1
    W16[0, 7] = 1
    
    # Factor 9: A02 B21 -> C01
    U16[2, 8] = 1
    V16[7, 8] = 1
    W16[2, 8] = 1
    
    # Factor 10: A12 B20 -> C10
    U16[5, 9] = 1
    V16[6, 9] = 1
    W16[1, 9] = 1
    
    # Factor 11: A12 B21 -> C11
    U16[5, 10] = 1
    V16[7, 10] = 1
    W16[3, 10] = 1
    
    # Factor 12: A00 B02 -> C02
    U16[0, 11] = 1
    V16[2, 11] = 1
    W16[4, 11] = 1
    
    # Factor 13: A01 B12 -> C02
    U16[1, 12] = 1
    V16[5, 12] = 1
    W16[4, 12] = 1
    
    # Factor 14: A02 B22 -> C02
    U16[2, 13] = 1
    V16[8, 13] = 1
    W16[4, 13] = 1
    
    # Factor 15: A10 B02 -> C12
    U16[3, 14] = 1
    V16[2, 14] = 1
    W16[5, 14] = 1
    
    # Factor 16: A11 B12 -> C12
    U16[4, 15] = 1
    V16[5, 15] = 1
    W16[5, 15] = 1
    
    # We're missing: A12 B22 -> C12
    # This is the 17th term we need to cover.
    
    # Check if rank 16 works as is (it won't, since we're missing one term)
    if _verify(U16, V16, W16, n, m, p, rank16):
        best_rank = rank16
        best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
    else:
        # Rank 16 with this structure doesn't work.
        # Try to find if we can modify one factor to cover two terms.
        
        # Idea: modify factor 16 to also cover A12*B22
        # Factor 16 currently: A11 B12 -> C12
        # We want: A11 B12 + A12 B22 -> C12
        # This would require U16[5, 15] = 1 (for A12) and V16[8, 15] = 1 (for B22)
        # But then we'd have U16[4, 15] = 1, U16[5, 15] = 1, V16[5, 15] = 1, V16[8, 15] = 1
        # The tensor contribution would be:
        # For (i=0, j=1, j'=1, k=2): U[1,15]*V[5,15]*W[5,15] = 1*1*1 = 1 (correct for A01*B12)
        # For (i=1, j=1, j'=1, k=2): U[4,15]*V[5,15]*W[5,15] = 1*1*1 = 1 (correct for A11*B12)
        # For (i=0, j=2, j'=2, k=2): U[2,15]*V[8,15]*W[5,15] = 0*1*1 = 0 (wrong, should be 0 for A02*B22 in C02)
        # For (i=1, j=2, j'=2, k=2): U[5,15]*V[8,15]*W[5,15] = 1*1*1 = 1 (correct for A12*B22 in C12)
        # But we also need to check that we don't introduce errors elsewhere.
        
        # Let's try this modification and verify
        U16_mod = U16.copy()
        V16_mod = V16.copy()
        W16_mod = W16.copy()
        
        U16_mod[5, 15] = 1  # Add A12
        V16_mod[8, 15] = 1  # Add B22
        
        if _verify(U16_mod, V16_mod, W16_mod, n, m, p, rank16):
            best_rank = rank16
            best_U, best_V, best_W = U16_mod.copy(), V16_mod.copy(), W16_mod.copy()
        else:
            # Try other modifications
            # What if we modify factor 14 (A10 B02 -> C12) to also cover something?
            # Or factor 15 (A11 B12 -> C12)?
            
            # Let's try modifying factor 15 to cover both A11*B12 and A12*B22
            U16_mod2 = U16.copy()
            V16_mod2 = V16.copy()
            W16_mod2 = W16.copy()
            
            # Factor 15: A11 B12 -> C12, add A12 B22
            U16_mod2[5, 15] = 1  # A12
            V16_mod2[8, 15] = 1  # B22
            # W16_mod2[5, 15] is already 1 (C12)
            
            if _verify(U16_mod2, V16_mod2, W16_mod2, n, m, p, rank16):
                best_rank = rank16
                best_U, best_V, best_W = U16_mod2.copy(), V16_mod2.copy(), W16_mod2.copy()
            else:
                # Try yet another approach: modify a Strassen factor to absorb a schoolbook term
                # This is complex. Let's just fall back to rank 17.
                best_rank = rank17
                best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
        # Ultimate fallback to schoolbook
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
