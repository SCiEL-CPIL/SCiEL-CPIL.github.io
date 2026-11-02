import numpy as np
import time
import sys
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # For each (i,j,k), compute sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i,j,:], Vr[j,k,:]*Wr[k,i,:])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 130  # Leave 20s safety margin before 150s kill
    
    # Strategy: Try to find a decomposition of rank < schoolbook_rank
    # using structured approaches: Strassen-like block reductions.
    
    # For 3x4x6, we can try to decompose into smaller blocks and apply
    # Strassen-like reductions within blocks.
    
    # Key insight: We can view the 3x4x6 multiplication as:
    # - Split A (3x4) into row blocks: A1 (2x4), A2 (1x4)
    # - Split B (4x6) into column blocks: B1 (4x4), B2 (4x2)
    # - Result: C11 = A1*B1 (2x4), C12 = A1*B2 (2x2), C21 = A2*B1 (1x4), C22 = A2*B2 (1x2)
    
    # Each sub-problem can potentially use Strassen-like reductions.
    
    # Let's try a more systematic approach:
    # 1. Try to find rank-71 by checking if one schoolbook term is redundant
    # 2. Try block-wise Strassen reductions
    
    # Approach 1: Check for linear dependencies in the schoolbook decomposition
    # The multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k).
    # A rank-R decomposition exists iff we can find U, V, W of rank R such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k).
    
    # For rank 71, we need to show that one of the 72 basis tensors is in the
    # span of the others. This is generally not the case for multiplication tensors.
    
    # Approach 2: Block-wise Strassen
    # For 2x2x2 blocks, Strassen gives rank 7 instead of 8.
    # Let's see if we can apply this to our 3x4x6 case.
    
    # Decompose into 2x2x2 blocks where possible:
    # A is 3x4, B is 4x6, C is 3x6
    # We can partition as:
    #   A = [A11 A12; A21 A22; A31 A32] where Aij are 1x2 or 2x2
    #   B = [B11 B12 B13; B21 B22 B23; B31 B32 B33; B41 B42 B43]
    
    # This is getting complex. Let's try a direct computational approach.
    
    # Method: Use the fact that for small matrices, we can try to find
    # explicit low-rank decompositions by solving the polynomial system.
    
    # For rank R, we have R*(n*m + m*p + p*n) unknowns and n*m*p equations.
    # For R=71: 71*(12+24+18) = 71*54 = 3834 unknowns, 72 equations.
    # This is highly underdetermined, so solutions likely exist if they exist at all.
    
    # The question is whether the equations are consistent.
    
    # Let's try a different approach: use the "subspace" method.
    # For each (i,j), define the vector v_{ij} in R^R where v_{ij}[r] = U[i*m+j,r].
    # Similarly for V and W.
    
    # Actually, let's just try to verify if rank 71 is possible by attempting
    # to construct it explicitly.
    
    # For rank 71, we can try: take the schoolbook 72 terms and see if we can
    # merge two of them into one using a Strassen-like trick.
    
    # Consider two terms (i1,j1,k1) and (i2,j2,k2). Can we replace them with
    # a single term that produces the same result?
    
    # This requires: U1[i1*m+j1]*V1[j1*p+k1]*W1[k1*n+i1] + 
    #                U2[i2*m+j2]*V2[j2*p+k2]*W2[k2*n+i2] = 1
    # while all other entries remain correct.
    
    # This is generally not possible without affecting other entries.
    
    # Let's try a more practical approach: use random search with better
    # initialization for rank 71.
    
    def try_rank_structured(R, n, m, p, num_trials=200, time_budget=8.0):
        """Try to find a rank-R decomposition using structured random search."""
        start = time.time()
        
        # Precompute indices
        ij_indices = [(i, j) for i in range(n) for j in range(m)]
        jk_indices = [(j, k) for j in range(m) for k in range(p)]
        ki_indices = [(k, i) for k in range(p) for i in range(n)]
        
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            # Start from schoolbook and perturb
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # First R-1 columns: use schoolbook terms
            for r in range(min(R - 1, n * m * p)):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            
            # Last column: random
            if R > 1:
                r = R - 1
                U[:, r] = np.random.randn(n * m) * 0.5
                V[:, r] = np.random.randn(m * p) * 0.5
                W[:, r] = np.random.randn(p * n) * 0.5
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try rank 71
    if time.time() - start_time < time_limit - 30:
        result = try_rank_structured(71, n, m, p, num_trials=100, time_budget=5.0)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 71
    
    # If we haven't improved, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
