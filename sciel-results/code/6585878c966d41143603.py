"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Start with schoolbook
    rank_school = n * m * p  # 120
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition of matrix mult tensor."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0])
        # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Actually the tensor is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T has entries T[i,j,k] = 1 for all valid i,j,k
        # because C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # A rank-r decomposition writes T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c is a vector of size n*m (indexed by i,j), 
        #       v_c is a vector of size m*p (indexed by j,k),
        #       w_c is a vector of size p*n (indexed by k,i)  [trace convention W[k*n+i,r]]
        #
        # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And we need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to find a better decomposition
    # Known bounds: for 4x5*5x6, schoolbook is 120
    # The minimum rank is at least max(n*m, m*p, p*n) = max(20,30,24) = 30
    # And at most n*m*p = 120
    
    # Let's try to use known results or construct a better one
    # For rectangular matrix multiplication, there are various known bounds.
    # 
    # One approach: try to find a decomposition of rank r < 120
    # using randomized or structured methods.
    
    # Let's try a simple improvement: if we can factor some terms
    # Actually, let me try to use the fact that for matrix multiplication,
    # we can potentially do better than schoolbook using Strassen-like methods
    # adapted to rectangular cases.
    
    # For now, let's try to search for a lower rank using a simple algorithm
    # We'll try ranks from some lower bound up to schoolbook - 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try a simple structured approach:
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

    # This is equivalent to finding matrices U, V, W such that
    # for each j, the n x p matrix U_j ⊗ V_j ⊗ W_j (in some sense) sums to all-ones

    # Actually, let me think differently. The matrix multiplication tensor can be written as:
    # T = sum_j T_j where T_j[i,k] = A[i,j]*B[j,k]
    # Each T_j is a rank-1 tensor (outer product of e_i and e_k... no)

    # Let me try a different approach: use the fact that the problem is equivalent to
    # finding the tensor rank of the matrix multiplication tensor for these dimensions.

    # For small dimensions, we can try to use integer programming or search.
    # Given time constraints, let's try a few heuristics.

    # Heuristic 1: Try to find a decomposition where we share some rank terms
    # between different (i,j,k) triples.

    # Let's try to construct a decomposition of rank less than schoolbook.
    # One known technique: if we can find orthogonal or sparse structures.

    # Actually, for this bootstrap, let me just verify the schoolbook works and return it.
    # The search for better decompositions would be complex.

    # Let me try one simple improvement: check if a slightly lower rank works
    # by trying to merge some terms.

    # Try rank = schoolbook - 1, -2, etc. with random perturbations
    random.seed(42)

    for trial_rank in range(best_rank - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() > deadline:
            break
        if trial_rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        # Use a simple randomized approach
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse matrices
            U = [[0] * trial_rank for _ in range(n * m)]
            V = [[0] * trial_rank for _ in range(m * p)]
            W = [[0] * trial_rank for _ in range(p * n)]
            
            # Try to assign each (i,j,k) to a random column
            assignments = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        assignments[(i, j, k)] = random.randint(0, trial_rank - 1)
            
            # This simple assignment won't work because we need the products to sum to 1
            # Let me try a different approach
            
            # Actually, let's use a structured approach:
            # For each j, we need the n x p sub-tensor to be all ones.
            # The j-th "slice" is U_j (n vectors) ⊗ V_j (p vectors) ⊗ W_j (n*p... no)
            
            # Let me reconsider the structure.
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            #
            # For fixed j, define:
            #   u_c[i] = U[i*m+j, c] for i in [0,n)
            #   v_c[k] = V[j*p+k, c] for k in [0,p)  
            #   w_c[k,i] = W[k*n+i, c] for k in [0,p), i in [0,n)
            #
            # Then T[i,j,k] = sum_c u_c[i] * v_c[k] * w_c[k,i]
            #
            # For fixed j, we need this to equal 1 for all i,k.
            
            # This is getting complex. Let me just return schoolbook for now.
            pass
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
