"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook decomposition is the baseline.
    # Rank = n * m * p = 120.
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions via random search
    # Strategy: try ranks from 119 down, with limited attempts per rank
    # Use random integer entries in {-1, 0, 1}
    
    np.random.seed(42)
    
    # For each candidate rank, try a limited number of random decompositions
    for rank in range(best_rank - 1, max(50, best_rank - 40), -1):
        if time.time() > deadline - 20:
            break
            
        max_attempts = 50 if rank > 100 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Generate random factor matrices with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(num_U, rank))
            V = np.random.choice([-1, 0, 1], size=(num_V, rank))
            W = np.random.choice([-1, 0, 1], size=(num_W, rank))
            
            # Check if this is a valid decomposition
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # must equal 1 for all (i,j,k)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1
                        actual = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]
                        # Wait, that's not right. The decomposition is:
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # This is a trilinear form, not a bilinear one.
                        # Actually, for matrix multiplication tensor decomposition:
                        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                        # The tensor is T[i,j,k] = delta (it's 1 for each triple)
                        # The decomposition means:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                        # 
                        # But wait, this is a 3-tensor, and the standard matrix multiplication
                        # tensor decomposition uses:
                        # (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # Hmm, that doesn't look right either. Let me reconsider.
                        #
                        # Actually, the standard way to view matrix multiplication as a tensor:
                        # The multiplication tensor T has indices (i, j, k) where i is row of A,
                        # j is the shared dimension, k is column of B.
                        # T[i,j,k] = 1 for all valid indices.
                        # A rank-r decomposition means:
                        # T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        #
                        # But the problem statement says U has shape (n*m, rank), V has shape (m*p, rank),
                        # W has shape (p*n, rank). So:
                        # U[i*m+j, r] corresponds to the (i,j) entry
                        # V[j*p+k, r] corresponds to the (j,k) entry  
                        # W[k*n+i, r] corresponds to the (k,i) entry
                        #
                        # And the decomposition is:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                        #
                        # This is a 3-way tensor decomposition. Let me verify the schoolbook:
                        # For schoolbook, rank = n*m*p = 120
                        # Each term r corresponds to a unique (i,j,k) triple
                        # U[i*m+j, r] = 1 if r corresponds to (i,j,k'), else 0
                        # V[j*p+k, r] = 1 if r corresponds to (i',j,k), else 0
                        # W[k*n+i, r] = 1 if r corresponds to (i',j',k) with i'=i, else 0
                        #
                        # Wait, that's not quite right. Let me re-examine the schoolbook code:
                        # for i in range(n):
                        #     for j in range(m):
                        #         for k in range(p):
                        #             U[i*m+j, col] = 1
                        #             V[j*p+k, col] = 1
                        #             W[k*n+i, col] = 1
                        #             col += 1
                        #
                        # So for each (i,j,k), we have a column where:
                        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
                        # All other entries in that column are 0.
                        #
                        # Then for a given (i,j,k):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # The only r that contributes is the one where (i,j,k) matches exactly.
                        # For that r: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
                        # So the sum is 1. Correct!
                        #
                        # For a different (i',j',k'):
                        # The r corresponding to (i,j,k) has:
                        # U[i'*m+j', r] = 0 unless i'=i and j'=j
                        # V[j'*p+k', r] = 0 unless j'=j and k'=k
                        # W[k'*n+i', r] = 0 unless k'=k and i'=i
                        # So if any of i'!=i, j'!=j, or k'!=k, at least one factor is 0.
                        # Correct!
                        #
                        # So the decomposition condition is:
                        # For all i in [0,n), j in [0,m), k in [0,p):
                        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        #
                        # Let me implement this check properly.
                        
                        break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check all (i,j,k) triples
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break  # Found a valid decomposition at this rank, try lower
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
