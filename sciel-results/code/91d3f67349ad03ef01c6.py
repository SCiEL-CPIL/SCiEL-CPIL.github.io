"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 112
    
    # Build the matrix multiplication tensor T[i,j,k] = e_{i,k} ⊗ e_j
    # We want to find rank-r decomposition: T = sum_{c=0}^{r-1} U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    
    # The tensor can be reshaped: for each (i,j,k), the entry at (i*m+j, j*p+k, k*n+i) is 1
    # We can represent this as a matrix of shape (n*m, m*p) for each "slice" indexed by... 
    # Actually, let's think of it differently.
    
    # The matrix multiplication tensor M_{i,j,k} has M[i,j,k] = delta_{i,k} (in some sense)
    # More precisely: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T[i,j,k] = 1 if we're multiplying, i.e., T is the structural tensor.
    
    # For decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} * (something)
    # Actually the structural tensor for matrix mult is: T[i,j,k] = 1 for all i,j,k (it's the "multiply" operation)
    # Wait, no. The matrix multiplication tensor T has entries T[i,j,k] = 1, and the multiplication is:
    # (A*B)_{i,k} = sum_{j} sum_{c} U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i,c]
    
    # Let me be more careful. The standard formulation:
    # We want: (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # A rank-r decomposition means:
    # (A*B)_{i,k} = sum_{c=0}^{r-1} (sum_{j} U[i*m+j,c] * A[i,j]) * (sum_{j} V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    
    # So the condition is: for all i,j,k:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  (this is the structural tensor entry)
    
    # Actually that's not quite right either. Let me think again.
    
    # The bilinear map for matrix multiplication:
    # f(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # A rank-r decomposition of the bilinear map means:
    # f(A,B)_{i,k} = sum_{c=0}^{r-1} L_c(A) * M_c(B) * W_{i,k,c}
    # where L_c(A) = sum_{i,j} U[i*m+j,c] * A_{i,j}
    # M_c(B) = sum_{j,k} V[j*p+k,c] * B_{j,k}
    
    # So: sum_j A_{i,j} B_{j,k} = sum_c (sum_{i',j'} U[i'*m+j',c] A_{i',j'}) * (sum_{j',k'} V[j'*p+k',c] B_{j',k'}) * W[k*n+i,c]
    
    # For this to hold for all A,B, we need:
    # For each (i,k): sum_j delta_{i,i'} delta_{j,j'} delta_{j,k'} ... 
    
    # The structural tensor condition is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all i,j,k
    
    # This is the condition we need to satisfy.
    
    # Let's verify with schoolbook: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c=(i,j,k)
    # sum over c: only c=(i,j,k) contributes: 1*1*1 = 1. Correct.
    
    # Now, can we do better than 112?
    # Known bounds: for 4x4 * 4x7, the rank is between... 
    # The exponent of matrix multiplication gives bounds.
    # For 4x4x4, best known is 49 (Strassen-based).
    # For 4x4x7, we can pad to 4x4x8 and use that, or find specific decompositions.
    
    # Let's try a randomized approach: start with schoolbook, then try to merge terms.
    
    # Strategy: Use the fact that some decompositions can be found by random initialization + gradient descent
    # on the constraint sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # But this is a non-convex problem. Let's try a few approaches.
    
    # Approach 1: Try to find a decomposition with rank < 112 by random search
    # This is unlikely to work well for such a high-dimensional problem.
    
    # Approach 2: Use known results. The rank of 4x4x4 matrix multiplication is at most 49.
    # For 4x4x7, we can think of it as 4x4x4 + 4x4x3.
    # Rank(4x4x7) <= Rank(4x4x4) + Rank(4x4x3)
    # Rank(4x4x3) <= 4*4*3 = 48 (schoolbook) or better.
    
    # Actually, let's just try to construct a valid decomposition with the best rank we can find.
    # Given the time constraint, let's try a simple approach:
    
    # 1. Start with schoolbook (rank 112)
    # 2. Try to find a lower-rank decomposition using random restarts and local search
    
    # But local search on this problem is very hard. Let's try a different approach.
    
    # For 4x4x4, the best known decomposition has rank 49 (using Strassen's algorithm generalized).
    # We can embed this into 4x4x7 by adding schoolbook terms for the remaining 3 columns.
    
    # Let's construct a 49-rank decomposition for 4x4x4 and pad to 4x4x7.
    
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we can use a tensor product or a specific construction.
    
    # Actually, let's just use the schoolbook decomposition since finding a better one
    # in the time available is challenging. But let's try a few things.
    
    # Try: rank 112 schoolbook as baseline
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to improve by random perturbation and projection
    # This is a stretch, but let's try a simple local search
    
    # Actually, given the complexity, let's just return the schoolbook decomposition.
    # The problem is very hard to improve upon without sophisticated algorithms.
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
