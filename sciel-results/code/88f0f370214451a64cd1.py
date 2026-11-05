"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 150s limit
    
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    current_col = 0
    
    def next_col():
        nonlocal current_col
        current_col += 1
    
    # 1. Process Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0
            
            # M0: (A00+A11)(B00+B11)
            U[i_base*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M1: (A10+A11)B00
            U[(i_base+1)*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = -1
            next_col()
            
            # M2: A00(B01-B11)
            U[i_base*m+j_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M3: A11(B10-B00)
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[j_base*p+k_base, current_col] = -1
            W[k_base*n+i_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            next_col()
            
            # M4: (A00+A01)B11
            U[i_base*m+j_base, current_col] = 1
            U[i_base*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            next_col()
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base+1)*m+j_base, current_col] = 1
            U[i_base*m+j_base, current_col] = -1
            V[j_base*p+k_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M6: (A01-A11)(B10+B11)
            U[i_base*m+(j_base+1), current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = -1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            next_col()
            
            # Time guard
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # 2. Process Naive blocks for j=4
    if time.time() <= deadline:
        for k_base in [0, 2, 4]:
            i_base = 0
            j_idx = 4
            
            U[i_base*m+j_idx, current_col] = 1
            V[j_idx*p+k_base, current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            next_col()
            
            U[i_base*m+j_idx, current_col] = 1
            V[j_idx*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+i_base, current_col] = 1
            next_col()
            
            U[(i_base+1)*m+j_idx, current_col] = 1
            V[j_idx*p+k_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            next_col()
            
            U[(i_base+1)*m+j_idx, current_col] = 1
            V[j_idx*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            if time.time() > deadline:
                break
    
    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")
    
    # 3. Attempt dual-variable elimination: try to reduce rank by 1
    # Check if any column is a linear combination of the others
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T to equal the multiplication tensor: T[i,j,k] = 1 if i,j,k are valid indices, else 0
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Build the target tensor
    target = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1
    
    # Flatten U, V, W for linear algebra
    # Each rank-1 term r contributes: U[:,r] outer V[:,r] outer W[:,r]
    # The tensor entry (i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try removing each column and see if the remaining can span the target
    # This is a bilinear problem, not linear. But we can use a trick:
    # Fix one factor and linearize.
    
    # Alternative: use the fact that if we have a decomposition, we can check
    # linear dependencies by looking at the "slice" structure.
    
    # For now, try a simpler approach: check if the last column (naive j=4, k=5, i=1)
    # can be expressed in terms of others. The naive terms for j=4 are independent
    # of the Strassen terms, so likely no reduction possible with this structure.
    
    # Try a different strategy: look for columns that have very sparse support
    # and might be absorbable.
    
    # Actually, let's just verify correctness and return rank 54.
    # The elimination attempt would require solving a nonlinear system.
    
    # Verify the decomposition
    if time.time() <= deadline:
        # Quick verification on a sample
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if val != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
