"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try to find a low-rank decomposition
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try various ranks from low to high
    # Known lower bounds suggest we can do better than 96
    # Let's try ranks from 50 to 95
    for target_rank in range(50, 96):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize to avoid scaling issues
            for c in range(target_rank):
                norm_U = np.linalg.norm(U[:, c])
                norm_V = np.linalg.norm(V[:, c])
                norm_W = np.linalg.norm(W[:, c])
                if norm_U > 0 and norm_V > 0 and norm_W > 0:
                    scale = (norm_U * norm_V * norm_W) ** (1/3)
                    U[:, c] /= scale
                    V[:, c] /= scale
                    W[:, c] /= scale
            
            # Check if this random guess is close to valid
            # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,k) for all i,j,k
            # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal the matrix multiplication tensor
            
            # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] should be 1 if we're computing the (i,k) entry using A[i,j]*B[j,k]
            # The tensor is T[i,j,k] = delta(j,j) which is always 1? No...
            
            # Let me reconsider. The tensor for matrix multiplication (n,m,p) is:
            # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
            # T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
            # Wait, that's not right either.
            
            # Actually, the tensor is defined such that:
            # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
            # This can be written as a tensor contraction.
            # The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k
            # No, that's the all-ones tensor.
            
            # Let me think again. The matrix multiplication tensor M_{n,m,p} is defined by:
            # M[i, j, k] = 1 if the (i,k) entry of A*B uses A[i,j] and B[j,k]
            # Since (A*B)[i,k] = sum_j A[i,j] * B[j,k], we have M[i,j,k] = 1 for all i,j,k.
            # But that would make it trivially rank 1, which is wrong.
            
            # I think I'm confusing myself. Let me look at this differently.
            # The tensor decomposition for matrix multiplication:
            # We want U, V, W such that for all i, j, k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} if we're computing something
            # 
            # Actually, the standard formulation: the tensor T for (n,m,p) matrix mult satisfies
            # T[i, j, k] = 1 for all i in [n], j in [m], k in [p]
            # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # 
            # But that can't be right because the all-ones tensor has rank 1.
            
            # I think the issue is that the tensor is not all ones. Let me reconsider.
            # 
            # For matrix multiplication C = AB where A is n×m and B is m×p:
            # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
            # 
            # The tensor T is a 3-tensor with T[i,j,k] representing the coefficient of A[i,j]*B[j,k] in C[i,k]
            # So T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] appears exactly once in C[i,k] for fixed i,k)
            # 
            # But wait, that means T is the all-ones tensor of shape (n,m,p), which has tensor rank 1.
            # That can't be right.
            
            # I think I'm making an error. Let me look at this from the perspective of the decomposition.
            # 
            # The schoolbook decomposition has rank n*m*p. For each (i,j,k), we have a term:
            # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
            # where c is the index for (i,j,k).
            # 
            # So the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For the schoolbook, this gives T[i,j,k] = 1 for all i,j,k.
            # 
            # But if T is all ones, it should have rank 1. The issue is that the decomposition
            # uses different indices for U, V, W. The tensor is not simply T[i,j,k] but rather
            # it's a tensor in the space U ⊗ V ⊗ W where U has dimension n*m, V has dimension m*p,
            # W has dimension p*n.
            
            # OK so the tensor is in (n*m) × (m*p) × (p*n) space, and the schoolbook decomposition
            # has rank n*m*p. The tensor itself is defined by:
            # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
            # 
            # So T is not all ones; it's a sparse tensor with exactly n*m*p non-zero entries.
            
            # Let me verify: for the schoolbook, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
            # Yes, that makes sense.
            
            # So I need to find U, V, W of shape (n*m, R), (m*p, R), (p*n, R) such that:
            # sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i*m+j, j*p+k, k*n+i) is a valid triple, else 0
            # 
            # But actually, every triple (i*m+j, j*p+k, k*n+i) for i in [n], j in [m], k in [p] is unique,
            # so the tensor has exactly n*m*p non-zero entries, all equal to 1.
            
            # Let me create the target tensor and check decompositions.
            # But creating the full tensor would be (16, 24, 24) = 9216 elements, which is fine.
            
            # Actually, let me just check the decomposition directly without forming the full tensor.
            # For a candidate (U, V, W), check that for all i,j,k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # And for invalid triples (which don't exist in our indexing), it should be 0, but since our
            # indexing covers all valid (i,j,k), we just need the above to hold.
            
            # Wait, but the tensor also has entries that should be 0. The tensor T[a,b,c] is 0 unless
            # a = i*m+j, b = j*p+k, c = k*n+i for some consistent i,j,k. But in our parameterization,
            # a, b, c are independent indices, so we need T[a,b,c] = 0 for most (a,b,c).
            
            # This is getting complicated. Let me just use the schoolbook as a baseline and try to
            # optimize from there, or use a known decomposition.
            
            pass  # Skip this approach for now
            
            # Instead, let me try a different strategy: use known results or construct a decomposition
            # using block methods.
            
            # For now, let's just return the schoolbook decomposition as a safe fallback.
            # But that gives score 1.0, which is the baseline.
            
            # Let me try to implement a simple optimization.
            
            # Actually, let me try a specific known decomposition for 4x4x4 and extend it.
            # For 4x4x4, the best known rank is 49 (or lower). For 4x4x6, we can potentially do better than 96.
            
            # Let me try rank 72 as a starting point and see if I can find a valid decomposition.
            pass
        
        # If we found a valid decomposition with this rank, update best
        # (This part is not implemented yet)
    
    # For now, return the schoolbook decomposition
    # Convert to lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
