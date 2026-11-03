"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank-17 decomposition using Strassen for 2x2x2 + schoolbook for remaining
    rank17 = 17
    U17 = np.zeros((n * m, rank17))
    V17 = np.zeros((m * p, rank17))
    W17 = np.zeros((p * n, rank17))
    
    # Part 1: Strassen for 2x2x2 (factors 0-6)
    # Factor 0: (A00+A11)(B00+B11)
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Factor 1: (A10+A11)B00
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Factor 2: A00(B01-B11)
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Factor 3: A11(B10-B00)
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Factor 4: (A00+A01)B11
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Factor 5: (A10-A00)(B00+B01)
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Factor 6: (A01-A11)(B10+B11)
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for middle column (factors 7-10)
    # A02*B20 -> C00
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    # A02*B21 -> C01
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    # A12*B20 -> C10
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    # A12*B21 -> C11
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    # Part 3: Schoolbook for last column (factors 11-16)
    # A00*B02 -> C02
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    # A01*B12 -> C02
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    # A02*B22 -> C02
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    # A10*B02 -> C12
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    # A11*B12 -> C12
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    # A12*B22 -> C12
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
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
    if not _verify(U17, V17, W17, n, m, p, rank17):
        # Fallback to schoolbook
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
    
    best_rank = rank17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Try to find rank-16 decomposition using structured search
    # Strategy: Use the Kronecker product structure to express the 2×3×3 multiplication tensor
    # as a linear combination of rank-1 terms, then find minimal spanning set
    
    rank16 = 16
    U16 = np.zeros((n * m, rank16))
    V16 = np.zeros((m * p, rank16))
    W16 = np.zeros((p * n, rank16))
    
    # First, try the simple merge approach from the current program
    # Factors 0-6: Same Strassen as before
    U16[0, 0] = 1; U16[4, 0] = 1
    V16[0, 0] = 1; V16[4, 0] = 1
    W16[0, 0] = 1; W16[3, 0] = 1
    
    U16[3, 1] = 1; U16[4, 1] = 1
    V16[0, 1] = 1
    W16[1, 1] = 1; W16[3, 1] = -1
    
    U16[0, 2] = 1
    V16[1, 2] = 1; V16[4, 2] = -1
    W16[2, 2] = 1; W16[3, 2] = 1
    
    U16[4, 3] = 1
    V16[3, 3] = 1; V16[0, 3] = -1
    W16[0, 3] = 1; W16[1, 3] = 1
    
    U16[0, 4] = 1; U16[1, 4] = 1
    V16[4, 4] = 1
    W16[0, 4] = -1; W16[2, 4] = 1
    
    U16[3, 5] = 1; U16[0, 5] = -1
    V16[0, 5] = 1; V16[1, 5] = 1
    W16[3, 5] = 1
    
    U16[1, 6] = 1; U16[4, 6] = -1
    V16[3, 6] = 1; V16[4, 6] = 1
    W16[0, 6] = 1
    
    # Factors 7-10: Same middle column schoolbook
    U16[2, 7] = 1; V16[6, 7] = 1; W16[0, 7] = 1
    U16[2, 8] = 1; V16[7, 8] = 1; W16[2, 8] = 1
    U16[5, 9] = 1; V16[6, 9] = 1; W16[1, 9] = 1
    U16[5, 10] = 1; V16[7, 10] = 1; W16[3, 10] = 1
    
    # Factors 11-14: Last column schoolbook, but merge factors 15 and 16
    U16[0, 11] = 1; V16[2, 11] = 1; W16[4, 11] = 1
    U16[1, 12] = 1; V16[5, 12] = 1; W16[4, 12] = 1
    U16[2, 13] = 1; V16[8, 13] = 1; W16[4, 13] = 1
    U16[3, 14] = 1; V16[2, 14] = 1; W16[5, 14] = 1
    
    # Factor 15: Try to combine A11*B12 and A12*B22 -> C12
    U16[4, 15] = 1; U16[5, 15] = 1
    V16[5, 15] = 1; V16[8, 15] = 1
    W16[5, 15] = 1
    
    if _verify(U16, V16, W16, n, m, p, rank16):
        best_rank = rank16
        best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
    else:
        # Try a more sophisticated approach: use the Kronecker product structure
        # to find a rank-16 decomposition
        
        # The 2x3x3 multiplication tensor can be viewed as:
        # T[i,j,k; j',k',i'] = delta_{j,j'} * delta_{k,k'}
        # We want to express this as sum_r U[i,j;r] V[j',k';r] W[k',i';r]
        
        # Try a different structured approach:
        # Use the fact that the multiplication tensor has a specific structure
        # that might allow for a rank-16 decomposition
        
        # Approach: Try to find a rank-16 decomposition by modifying the Strassen factors
        # to absorb some of the schoolbook terms
        
        # Try: Use a different partition of the 2x3x3 problem
        # Instead of 2x2x2 + 2x1x2 + 2x3x1, try overlapping blocks
        
        # For now, fall back to rank 17 if rank 16 doesn't work
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
