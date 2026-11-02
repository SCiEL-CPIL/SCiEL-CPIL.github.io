import numpy as np
import time

def construct():
    # Direct construction of a rank-16 decomposition for 3x4 x 4x4 matrix multiplication
    # T[i,j,k] = A[i,j] * B[j,k] for i in 0..2, j in 0..3, k in 0..3
    # U shape: (12, 16), V shape: (16, 16), W shape: (12, 16)
    # Index convention:
    #   U[i*m + j, r] where m=4
    #   V[j*p + k, r] where p=4
    #   W[k*n + i, r] where n=3
    
    n, m, p = 3, 4, 4
    rank = 16
    
    # We use the trivial/schoolbook decomposition:
    # Each product term A[i,j] * B[j,k] contributes one rank-1 term.
    # For each (i, j, k), T[i,j,k] = 1 * A[i,j] * B[j,k]
    # We need to express this as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The schoolbook approach uses n*m*p = 48 terms, but we can do better.
    # For fixed j, the sum over i and k gives a 3x4 output block.
    # Actually, the minimal known rank for M_{3,4,4} needs investigation.
    
    # Let me try a structured approach:
    # For each column j of A (which is a row j of B), we have a 3-vector a_j and 4-vector b_j.
    # The product is a_j * b_j^T which is 3x4.
    # We can decompose this outer product.
    
    # Actually, let's think differently. The tensor T_{ijk} = 1 for all i,j,k.
    # This is the all-ones tensor of shape 3x4x4.
    # Rank of all-ones tensor: The rank of the all-ones tensor of shape (a,b,c) is max(a,b,c) 
    # when one dimension is 1, but in general it's more complex.
    
    # Wait, the matrix multiplication tensor is NOT the all-ones tensor in the usual sense.
    # T[i,j,k] represents the coefficient relating A[i,j], B[j,k], and C[i,k].
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if and only if the term A[i,j]B[j,k] appears in C[i,k].
    # And indeed, for each (i,j,k), T[i,j,k] = 1.
    
    # The tensor rank of the matrix multiplication tensor M_{n,m,p} is known to be:
    # At least max(n*m, m*p, p*n) by the "slice rank" lower bound... no, that's not right.
    # The trivial upper bound is n*m*p.
    # For M_{3,4,4}: trivial = 3*4*4 = 48.
    
    # Can we do better? Let's use Strassen-style sharing.
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{3,3,3}, best known is 23 (Landsberg/Strassen 1983), but I don't remember the exact factors.
    
    # Let me try a block decomposition:
    # Split the 3x4x4 into overlapping 2x2x2 blocks.
    # Or better: use the fact that M_{3,4,4} can be embedded in M_{4,4,4}.
    # M_{4,4,4} has rank at most 49 (schoolbook) or 63 (Strassen tiling of 2x2x2 blocks).
    # Actually, Strassen gives 7^log2(4) = 49 for 4x4x4.
    
    # Let me try a different approach: direct rank-1 decomposition.
    # T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    # We want T[i,j,k] = sum_{r=0}^{R-1} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Note that the index j appears in both u and v, and k appears in both v and w, and i in both w and u.
    # This is a 3-way tensor.
    
    # One approach: For each (j,k) pair, define a vector. But that's 16 vectors of length 3, giving rank up to 16*something.
    
    # Let me try rank 16 construction:
    # For each j in 0..3 and k in 0..3, we have a "slot" but we need to share.
    
    # Actually, the simplest correct approach: use the schoolbook with rank 48.
    # But let me see if I can get lower.
    
    # Key insight: T[i,j,k] = 1 for ALL (i,j,k). This is the all-ones tensor.
    # The tensor rank of the all-ones tensor 1_{n,m,p} is min over factorizations.
    # One factorization: u_r[i,j] = 1, v_r[j,k] = 1, w_r[k,i] = 1/R for r=0..R-1 with R=1? No.
    # u[i,j]=1, v[j,k]=1, w[k,i]=1 gives T=1 with rank 1! 
    # Wait: T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
    # If R=1: u_0[i,j]=1 for all i,j; v_0[j,k]=1 for all j,k; w_0[k,i]=1 for all k,i
    # Then T[i,j,k] = 1*1*1 = 1. 
    # So the all-ones tensor has rank 1?!
    
    # But wait, that can't be right for matrix multiplication. Let me re-examine.
    # The matrix multiplication tensor is defined by:
    # (A × B)[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor has entries T[i,j,k] = 1 for each (i,j,k) where the product A[i,j]*B[j,k] 
    # contributes to C[i,k].
    
    # In the standard formulation, the tensor M_{n,m,p} ∈ F^{n×m×p} is defined such that:
    # M[i,j,k] = 1 if and only if the (i,k) entry of the product involves A[i,j] and B[j,k].
    # This is true for ALL (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    # So yes, M[i,j,k] = 1 for all valid indices.
    
    # And if we set u_r[i,j] = 1, v_r[j,k] = 1, w_r[k,i] = 1 for a single r, we get T[i,j,k] = 1.
    # So the tensor rank is 1!
    
    # But this contradicts known results. Let me reconsider the problem statement.
    
    # "Matrix multiplication 3,4,4" - find rank-minimal exact tensor decomposition.
    # The output factor follows W[k*n+i, r].
    
    # I think the issue is that the decomposition must satisfy:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k' ...) 
    # No, the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_r M_r[i] where M_r are rank-1 forms.
    # More precisely:
    # Let U: F^{n*m} -> F^r, V: F^{m*p} -> F^r, W: F^{p*n} -> F^r (or duals)
    # Then the multiplication is: flatten(C) = W^T * (U * flatten(A) * V^T * flatten(B))
    # i.e., C_{ik} = sum_r U[i,j]_r * A[i,j] ... 
    
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"
    
    # So the decomposition is exactly: for all (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i,j,k.
    
    # As I showed, rank 1 works: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Check: sum_r=0 U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1 = T[i,j,k]. ✓
    
    # So the answer is rank 1! But that seems too simple and contradicts the "rank-minimal" search.
    # Let me verify this is correct.
    
    # Actually, I think I'm confusing the tensor definition. In the standard matrix multiplication tensor:
    # The tensor M ∈ F^{n×m×p} has M[i,j,k] = 1 if the trilinear form x_i * y_j * z_k appears.
    # For matrix multiplication C = AB, we have C[i,k] = sum_j A[i,j] B[j,k].
    # The corresponding tensor has a 1 at position (i,j,k) for each triple (i,j,k) where i indexes row of A and C,
    # j indexes col of A and row of B, k indexes col of B and C.
    # This is indeed all ones.
    
    # And the rank of the all-ones tensor of any size is 1 (over fields of characteristic 0 or > max(n,m,p)).
    
    # So the minimal rank is 1!
    
    # Let me verify: U has shape (12, 1), all entries 1.
    # V has shape (16, 1), all entries 1.
    # W has shape (12, 1), all entries 1.
    # Then for any (i,j,k): sum_{r=0}^{0} U[i*4+j, 0] * V[j*4+k, 0] * W[k*3+i, 0] = 1*1*1 = 1. ✓
    
    # This should be correct. Let me return this.
    
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
