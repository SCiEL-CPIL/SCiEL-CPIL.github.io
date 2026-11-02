import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 48
    
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    # Schoolbook solution (guaranteed correct)
    U_school = np.zeros((n * m, rank_school), dtype=np.int64)
    V_school = np.zeros((m * p, rank_school), dtype=np.int64)
    W_school = np.zeros((p * n, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_school
    
    # Try to find a rank 47 solution by exploiting structure
    # Strategy: For 3x4 * 4x4, we have 12 outputs, each is sum of 4 products.
    # Total schoolbook: 48 multiplications.
    # Try to merge one multiplication across multiple outputs.
    
    if time.time() < deadline - 10:
        # Attempt rank 47: try to find if one pair (j,k) can serve multiple (i) outputs
        # with appropriate coefficients, saving one multiplication.
        
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For schoolbook, each (i,j,k) has exactly one r where all three are 1.
        # To get rank 47, we need to combine two (i,j,k) triples into one rank-1 term
        # such that the bilinear form still works.
        
        # This requires: for some (i1,j,k) and (i2,j,k) with i1 != i2,
        # we can have U[i1*m+j, r] = a, U[i2*m+j, r] = b,
        # V[j*p+k, r] = 1,
        # W[k*n+i1, r] = 1/a, W[k*n+i2, r] = 1/b
        # But this would affect other terms unless a, b are integers/half-integers
        # and the cross terms cancel or are zero.
        
        # Actually, for exact integer decomposition, we need a, b to be integers
        # and 1/a, 1/b to be integers or half-integers, which is very restrictive.
        
        # Alternative: use the fact that for rectangular matrices, we might be able
        # to use a "partial Strassen" approach.
        
        # Let's try a different approach: check if rank 47 is possible by solving
        # a linear system with 47 rank-1 terms.
        
        # The multiplication tensor has 3*4*4 = 48 nonzero entries.
        # A rank-r decomposition requires:
        # sum_{r=1}^{R} u_r[i,j] * v_r[j,k] * w_r[k,i] = delta_{i,j,k}
        
        # For R=47, we need to find 47 triples (u_r, v_r, w_r) such that
        # the sum equals the tensor.
        
        # This is a hard problem. Let's try a heuristic: start with schoolbook
        # and try to merge two terms.
        
        # Try merging (i=0, j=0, k=0) and (i=1, j=0, k=0) into one term
        # U[0, r] = 1, U[4, r] = 1  (i=0,j=0 and i=1,j=0)
        # V[0, r] = 1  (j=0, k=0)
        # W[0, r] = 1, W[3, r] = 1  (k=0, i=0 and k=0, i=1)
        
        # Then for output (0,0): contribution from this term is 1*1*1 = 1 (correct)
        # For output (1,0): contribution from this term is 1*1*1 = 1 (correct)
        # But this term also contributes to other outputs if U, V, W have other nonzero entries.
        
        # The problem is that U[0, r] = 1 means this term contributes to ALL outputs (i, j=0)
        # via W[k*n+i, r]. So if W has nonzero entries at multiple k*n+i positions,
        # we get unwanted contributions.
        
        # For this to work, we need W to be nonzero only at k=0, i=0 and k=0, i=1.
        # Similarly, V must be nonzero only at j=0, k=0.
        # And U must be nonzero only at i=0, j=0 and i=1, j=0.
        
        # Let's check: if we set:
        # U[0, r] = 1, U[4, r] = 1, all other U[*, r] = 0
        # V[0, r] = 1, all other V[*, r] = 0
        # W[0, r] = 1, W[3, r] = 1, all other W[*, r] = 0
        
        # Then the contribution to output (i, j, k) is:
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This is nonzero only if:
        # i*m+j in {0, 4} => (i=0,j=0) or (i=1,j=0)
        # j*p+k = 0 => (j=0, k=0)
        # k*n+i in {0, 3} => (k=0,i=0) or (k=0,i=1)
        
        # So nonzero contributions occur for:
        # (i=0,j=0,k=0): U[0]=1, V[0]=1, W[0]=1 => 1
        # (i=1,j=0,k=0): U[4]=1, V[0]=1, W[3]=1 => 1
        
        # These are exactly the two terms we wanted to merge!
        # And there are no other nonzero contributions.
        
        # So we can replace the two schoolbook terms for (0,0,0) and (1,0,0)
        # with a single merged term, reducing rank by 1!
        
        # Let's verify this doesn't break anything else.
        # The merged term only contributes to outputs (0,0,0) and (1,0,0),
        # which is exactly what we need.
        
        # So rank 47 should be achievable!
        
        U_47 = np.zeros((n * m, 47), dtype=np.int64)
        V_47 = np.zeros((m * p, 47), dtype=np.int64)
        W_47 = np.zeros((p * n, 47), dtype=np.int64)
        
        # Fill with schoolbook terms, skipping (i=0,j=0,k=0) and (i=1,j=0,k=0)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i == 0 and j == 0 and k == 0) or (i == 1 and j == 0 and k == 0):
                        continue
                    U_47[i * m + j, col] = 1
                    V_47[j * p + k, col] = 1
                    W_47[k * n + i, col] = 1
                    col += 1
        
        # Add the merged term at column 46
        U_47[0, 46] = 1   # i=0, j=0
        U_47[4, 46] = 1   # i=1, j=0
        V_47[0, 46] = 1   # j=0, k=0
        W_47[0, 46] = 1   # k=0, i=0
        W_47[3, 46] = 1   # k=0, i=1
        
        # Verify the decomposition
        def verify_decomp(U, V, W, n, m, p):
            R = U.shape[1]
            # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            # for all i,j,k
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(R):
                            total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if total != 1:
                            return False
            return True
        
        if verify_decomp(U_47, V_47, W_47, n, m, p):
            best_U = U_47.tolist()
            best_V = V_47.tolist()
            best_W = W_47.tolist()
            best_rank = 47
            print(f"Found rank 47 decomposition!")
        else:
            print("Rank 47 verification failed, using schoolbook rank 48")
    
    return {"U": best_U, "V": best_V, "W": best_W}
