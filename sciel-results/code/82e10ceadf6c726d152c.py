import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Tensor T_{i,j,k} = delta_{i,k} for A[i,j] * B[j,k]
    # We need U, V, W such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
    
    # Start with the rank-42 Strassen decomposition, then try to reduce rank
    # by finding linear dependencies in the factor matrices.
    
    # First, build the rank-42 decomposition as before
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Now try to reduce rank by finding linear dependencies
    # The tensor is T_{i,j,k} = delta_{i,k}
    # We can try to use the fact that the problem has structure
    
    # Alternative approach: build a minimal decomposition directly
    # For M_{2,4,6}, the tensor has 2*4*6 = 48 entries
    # The rank is at least max(2*4, 4*6, 2*6) = 24 (trivial lower bound)
    # But due to the delta structure, we can do better
    
    # Let's try a direct construction
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k}
    
    # Strategy: For each (i,k) pair where i != k, we need the sum to be 0
    # For (i,i) pairs, we need the sum to be 1
    
    # Let's use a different approach: express the multiplication as a sum of rank-1 terms
    # where each term is U_r * V_r * W_r (outer products)
    
    # Actually, let me try to find a rank-24 or rank-28 decomposition
    # by exploiting the block structure more carefully
    
    # For a 2x4 times 4x6:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
    # ...
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] + A[1,3]*B[3,0]
    # ...
    
    # The key insight: we can share computation between the two rows of C
    # Let's define:
    # S0 = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:] + A[0,3]*B[3,:]  (row 0 of C)
    # S1 = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:] + A[1,3]*B[3,:]  (row 1 of C)
    
    # Each S has 6 entries, so naively we need 12 multiplications
    # But we can use Strassen-like tricks on the 4x4 blocks
    
    # Let me try a rank-28 approach using a hybrid method
    # Split the 4 columns into two groups of 2
    
    # Group 1: columns 0,1
    # Group 2: columns 2,3
    
    # For each group, use a 2x2x6 approach
    
    # Actually, let me try to be more systematic
    # The tensor M_{2,4,6} can be seen as two copies of M_{2,2,6} (for the two column groups)
    # Plus cross terms... no, that's not right.
    
    # Let me just return the rank-42 solution for now, but with a time guard
    t0 = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Try to find a better decomposition using linear algebra
    # Build the tensor as a 3D array
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1 if i == k else 0
    
    # Flatten to get the tensor in matrix form
    # The matrix multiplication tensor can be represented as:
    # For each (i,j), we have a vector in R^p: T[i,j,:] = e_i (standard basis vector)
    # So T[i,j,k] = delta_{i,k}
    
    # The rank is the minimum number of rank-1 tensors that sum to T
    # A rank-1 tensor has the form u_i * v_j * w_k
    
    # Let's try to find a decomposition with rank < 42
    # by solving a linear system
    
    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}
    
    # This is a non-linear system. Let's try a different approach.
    
    # We can use the fact that the tensor is sparse (only n*p = 12 non-zero entries)
    # to construct a sparse decomposition.
    
    # For each (i,k) with i==k, we need:
    # sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # For each (i,k) with i!=k, we need:
    # sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 0
    
    # Let's try a rank-24 construction:
    # Use 4 "multipliers" per (i,k) diagonal element, but share them
    
    # Actually, the simplest correct answer is rank 42.
    # Let me verify the current decomposition is correct and return it.
    
    # Verify
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                expected = 1 if i == k else 0
                if s != expected:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # If invalid, return zeros (shouldn't happen)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
