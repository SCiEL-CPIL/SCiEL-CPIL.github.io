"""Matrix multiplication 2x4 by 4x8 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    
    # We want to decompose the matrix multiplication tensor T[i,j,k] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    # as sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Strategy: Use a block decomposition based on Strassen's algorithm.
    # We can partition the j dimension (size 4) into groups and k dimension (size 8) into groups.
    
    # Let's try a systematic approach:
    # Partition j = {0,1,2,3} into groups and k = {0,...,7} into groups.
    # For each (j_group, k_group) pair, we compute the sub-matrix product.
    
    # The simplest approach: treat it as 4 independent 2x1 * 1x8 products?
    # No, that doesn't work directly.
    
    # Let's use the standard Strassen-based approach:
    # Partition j into pairs: {0,1} and {2,3}
    # Partition k into pairs: {0,1}, {2,3}, {4,5}, {6,7}
    # For each (j_pair, k_pair), use Strassen 2x2*2x2 (rank 7)
    # Total: 2 * 4 * 7 = 56
    
    # But we can do better by using larger blocks or different decompositions.
    
    # Let's try: partition j into singletons and k into pairs.
    # For each j (1x1) and each k_pair (1x2): 1x1 * 1x2, rank 1
    # Total: 4 * 4 * 1 = 16? That seems too low.
    
    # Wait, let me reconsider. The tensor T[i,j,k] = delta_{i,j,k} for matrix mult
    # means T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index).
    
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the indices are consistent, which they always are.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # The rank of this tensor is what we're minimizing.
    
    # Let's build the 56-rank Strassen solution first, then try to improve.
    
    # Build Strassen 2x2*2x2 decomposition for each (j_pair, k_pair)
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    t = 0
    for a in range(2):  # j-group index: 0 or 1, covering j=2a, 2a+1
        for b in range(4):  # k-group index: 0..3, covering k=2b, 2b+1
            # Strassen 2x2*2x2: 7 terms
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10-A00)(B01)
            # M3 = (A00)(B10+B11)
            # M4 = (A11)(B00-B10)
            # M5 = (A00+A01)(B10)
            # M6 = (A10-A00)(B01)  -- wait, let me use the standard Strassen
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)(B00)
            # M3 = (A00)(B10+B11)
            # M4 = (A11)(B00-B10)
            # M5 = (A00+A01)(B10)
            # M6 = (A10-A00)(B01)
            # M7 = (A01-A11)(B00+B10)
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            # U contributions (A indices: row i, col j)
            # A00 = A[0, 2a], A01 = A[0, 2a+1], A10 = A[1, 2a], A11 = A[1, 2a+1]
            # U index: i*m + j
            
            # M1: (A00+A11)(B00+B11)
            U[0*m + 2*a, t] += 1  # A00
            U[1*m + 2*a+1, t] += 1  # A11
            V[2*a*p + 2*b, t] += 1  # B00
            V[(2*a+1)*p + 2*b+1, t] += 1  # B11
            W[2*b*n + 0, t] += 1  # C00
            W[(2*b+1)*n + 1, t] += 1  # C11
            t += 1
            
            # M2: (A10+A11)(B00)
            U[1*m + 2*a, t] += 1  # A10
            U[1*m + 2*a+1, t] += 1  # A11
            V[2*a*p + 2*b, t] += 1  # B00
            W[2*b*n + 1, t] += 1  # C10
            W[(2*b+1)*n + 1, t] += -1  # C11
            t += 1
            
            # M3: (A00)(B10+B11)
            U[0*m + 2*a, t] += 1  # A00
            V[(2*a+1)*p + 2*b, t] += 1  # B10
            V[(2*a+1)*p + 2*b+1, t] += 1  # B11
            W[(2*b+1)*n + 0, t] += 1  # C01
            W[(2*b+1)*n + 1, t] += 1  # C11
            t += 1
            
            # M4: (A11)(B00-B10)
            U[1*m + 2*a+1, t] += 1  # A11
            V[2*a*p + 2*b, t] += 1  # B00
            V[(2*a+1)*p + 2*b, t] += -1  # B10
            W[2*b*n + 0, t] += 1  # C00
            W[2*b*n + 1, t] += 1  # C10
            t += 1
            
            # M5: (A00+A01)(B10)
            U[0*m + 2*a, t] += 1  # A00
            U[0*m + 2*a+1, t] += 1  # A01
            V[(2*a+1)*p + 2*b, t] += 1  # B10
            W[2*b*n + 0, t] += -1  # C00
            W[(2*b+1)*n + 0, t] += 1  # C01
            t += 1
            
            # M6: (A10-A00)(B01)
            U[1*m + 2*a, t] += 1  # A10
            U[0*m + 2*a, t] += -1  # A00
            V[2*a*p + 2*b+1, t] += 1  # B01
            W[2*b*n + 0, t] += -1  # C00
            t += 1
            
            # M7: (A01-A11)(B00+B10)
            U[0*m + 2*a+1, t] += 1  # A01
            U[1*m + 2*a+1, t] += -1  # A11
            V[2*a*p + 2*b, t] += 1  # B00
            V[(2*a+1)*p + 2*b, t] += 1  # B10
            W[2*b*n + 0, t] += -1  # C00
            W[(2*b+1)*n + 1, t] += 1  # C11
            t += 1
    
    # Verify the solution
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for tt in range(rank):
                    rhs += U[i*m+j, tt] * V[j*p+k, tt] * W[k*n+i, tt]
                if abs(lhs - rhs) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: return schoolbook rank 64
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, t] = 1
                    V_sb[j*p+k, t] = 1
                    W_sb[k*n+i, t] = 1
                    t += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
