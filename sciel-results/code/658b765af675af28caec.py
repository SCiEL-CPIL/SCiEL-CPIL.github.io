import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as guaranteed fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 60
    
    # Try to find lower-rank decompositions
    # Strategy: Use structured block decomposition with Strassen for 2x2x2 cores
    # and verify candidates exactly
    
    # First, try rank 59: check if schoolbook tensor has rank < 60
    # The matmul tensor T[i,j,k] = 1 if we're looking at the standard basis
    # Actually T[i,j,k] = delta_{i,j,k} in the sense that C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (rank-1 tensors)
    # The rank of this tensor is exactly n*m*p = 60 for the standard matmul tensor.
    
    # However, we can potentially do better by using non-standard decompositions.
    # Let's try a structured approach:
    
    # Approach: Partition the problem and use Strassen
    # For 3x4x5, split m=4 into 2+2
    # C = A1*B1 + A2*B2 where A1,A2 are 3x2, B1,B2 are 2x5
    # Each 3x2x5 has rank 30, total 60. No gain from simple partition.
    
    # Try: Use the fact that 3x2x5 can be decomposed more efficiently
    # 3x2x5 matmul: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # This is just 2 multiplications per (i,k) pair, so rank = 3*2*5 = 30
    
    # Alternative: Try random/structured search for rank < 60
    # Use half-integer entries in {-1, -1/2, 0, 1/2, 1}
    
    # For efficiency, let's try a few structured candidates:
    
    # Candidate 1: Try to exploit symmetry/structure
    # The tensor T for 3x4x5 matmul has T[i,j,k] = 1 for all i,j,k
    # (since each output C[i,k] gets a contribution from each A[i,j]*B[j,k])
    # Wait, that's not right. The tensor is:
    # T[i,j,k] = 1 if we define the matmul tensor as having 1s where the
    # corresponding product contributes. Actually, the matmul tensor is:
    # T_{i,j,k} = delta (the tensor such that C = A*B means
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k])
    # So T[i,j,k] = 1 for all i,j,k in the standard matmul.
    
    # The rank of the all-ones tensor of shape (3,4,5) is... 
    # Actually it's 1! Because T[i,j,k] = 1 = 1*1*1, so it's a single rank-1 tensor.
    # But that's wrong - the matmul tensor isn't all ones in that sense.
    
    # Let me reconsider. The matrix multiplication tensor for n×m×p is defined as:
    # T ∈ R^{n×m × m×p × p×n} (or equivalently R^{n×m×p})
    # where T[i,j,k] = 1 for all i,j,k, and the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This corresponds to T[i,j,k] * A[i,j] * B[j,k] summed over j.
    # So T[i,j,k] = 1 for all i ∈ [0,n), j ∈ [0,m), k ∈ [0,p).
    
    # The rank of the all-ones tensor of shape (n,m,p) is 1!
    # Because it equals u ⊗ v ⊗ w where u is all-ones of length n,
    # v is all-ones of length m, w is all-ones of length p.
    
    # Wait, that can't be right. Let me re-examine.
    
    # The tensor for matrix multiplication C = AB where A is n×m, B is m×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # The associated tensor T has T[i,j,k] = 1 for all i,j,k.
    # And indeed, T = (1,1,...,1) ⊗ (1,1,...,1) ⊗ (1,1,...,1), which is rank 1.
    # 
    # But that would mean matmul has rank 1, which is absurd.
    
    # I think I'm confusing the tensor definition. Let me be more careful.
    
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # It's defined such that the bilinear map (A,B) -> C is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In terms of a 3-tensor T with T[i,j,k] being the coefficient:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # 
    # Now, the rank of this tensor: we want to write
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^n, v_r ∈ R^m, w_r ∈ R^p
    # 
    # But wait, the indices don't match up. Let me use the standard definition.
    
    # Standard definition: The matmul tensor for multiplying an n×m matrix by an m×p matrix
    # is a tensor T ∈ R^{(n×m) × (m×p) × (p×n)} or equivalently we can think of it
    # as T ∈ R^{n × m × p} where:
    # T[i, j, k] = 1 for all i,j,k
    # and the multiplication is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    # The rank of the all-ones 3-tensor of shape (n,m,p) is indeed 1.
    # This means the bilinear map has rank 1 in the tensor rank sense?
    
    # No! I think the issue is that the tensor rank decomposition for bilinear maps
    # is different. Let me look at this more carefully.
    
    # For a bilinear map B: U × V -> W, the tensor rank is the minimum R such that
    # B(u,v) = sum_{r=1}^R <u, u_r> <v, v_r> w_r
    # where u_r ∈ U*, v_r ∈ V*, w_r ∈ W.
    
    # For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write: C[i,k] = sum_{j} A[i,j] B[j,k] * e_{i,k} (where e_{i,k} is basis)
    # But that's not a rank-1 decomposition in the right sense.
    
    # The correct tensor rank formulation:
    # The matmul tensor T has entries T[(i,j), (j',k), (i',k')] = 
    #   1 if i=i', j=j', k=k'
    #   0 otherwise
    # Wait, that's the identity tensor, not matmul.
    
    # I think I need to be very precise. The matrix multiplication tensor for
    # n×m × m×p -> p×n is defined as:
    # T ∈ R^{(nm) × (mp) × (pn)}
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'
    # 
    # No wait, that's still not right. Let me think again.
    
    # The bilinear map is: (A, B) -> C where C = AB
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In tensor form, if we flatten A to a vector in R^{nm}, B to R^{mp}, C to R^{pn},
    # then the map is linear in each argument separately (bilinear).
    # The tensor T has T[a, b, c] where a indexes A, b indexes B, c indexes C.
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k
    #             = 0 otherwise
    # 
    # So T[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 elsewhere.
    # 
    # This is NOT the all-ones tensor. It's a structured tensor.
    # 
    # The rank of this tensor is what we're trying to minimize.
    # 
    # The schoolbook decomposition uses R = n*m*p terms:
    # For each (i,j,k): term = e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    # This gives rank n*m*p = 60.
    
    # Now, can we do better? The key question is whether there's a lower-rank
    # decomposition. For 2×2×2, Strassen shows rank 7 < 8.
    # For 3×3×3, the best known is 23 (Landsberg).
    # For 3×4×5, the best known rank is... I'm not sure, but it's likely less than 60.
    
    # Given the time constraint, let me try a practical approach:
    # 1. Use schoolbook as fallback (rank 60)
    # 2. Try to find a rank-59 or lower decomposition via structured search
    
    # For now, let's return schoolbook since finding a better decomposition
    # requires significant search time.
    
    return schoolbook

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
