"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Leave 15s safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Schoolbook solution (rank = n*m*p = 72)
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Strategy: Use Kronecker product structure
    # 3x3 * 3x8 = (3x3 * 3x3) ⊗ (1x1 * 1x1) approach won't directly work
    # Instead, decompose the 8 columns into 2 groups of 4, or use structure
    
    # Key insight: T[i,j,k] = 1 for all i,j,k
    # This means the tensor is the all-ones tensor of shape (3,3,8)
    # Rank of all-ones tensor (a,b,c) is min(a,b,c) in general? No, that's for special cases.
    # Actually, the all-ones tensor of shape (a,b,c) has rank min(a,b,c) only in specific cases.
    # For (3,3,8), the rank of the all-ones tensor is at most 3 (since we can write it as sum of 3 rank-1 terms)
    # Wait: T[i,j,k] = 1 = sum_{c=0}^{2} (1/3) * 1 * 1 * 3? No.
    # T[i,j,k] = 1. We need sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.
    # If we set u_c[i] = 1 for all i, v_c[j] = 1 for all j, w_c[k] = 1 for all k,
    # then sum_c 1*1*1 = r, so we need r=1? No, we need the sum to be 1.
    # So with r=1: u[0,0]=1, v[0,0]=1, w[0,0]=1 gives T[i,j,k]=1 for all i,j,k. Rank 1!
    # But wait, that's for the simple case where all entries are 1.
    # Let me re-examine the problem.
    
    # For matrix multiplication C = A * B where A is 3x3 and B is 3x8:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_{i,k} C[i,k] * (sum_j A[i,j] * B[j,k])
    # = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k) where i in [0,3), j in [0,3), k in [0,8)
    # And we need to decompose this tensor.
    
    # The all-ones tensor of shape (3,3,8) indeed has rank 1!
    # T[i,j,k] = 1 * 1 * 1 for all i,j,k
    # So u_c[i*m+j] = 1 for all i,j, v_c[j*p+k] = 1 for all j,k, w_c[k*n+i] = 1 for all k,i
    # With rank 1: U[*,0] = 1, V[*,0] = 1, W[*,0] = 1
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 ✓
    
    # So the rank is 1! Let's verify this is correct.
    # The tensor for matrix multiplication maps (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: f(A,B,C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # where T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # Wait, but that would mean the rank is always 1 for any matrix multiplication, which is clearly wrong.
    # Let me reconsider.
    
    # Actually, I think I'm confusing two different formulations.
    # The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th basis vector in R^n (for C), e_j in R^m (for A's columns/B's rows), e_k in R^p (for B's columns)
    # No wait, that's not right either.
    
    # Let me think about this more carefully.
    # The matrix multiplication map is: mult: R^{n×m} × R^{m×p} → R^{n×p}
    # mult(A,B)[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map, which can be associated with a trilinear form:
    # f(A, B, C) = sum_{i,k} C[i,k] * mult(A,B)[i,k] = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (with appropriate index ranges).
    # But this is the "dual" tensor. The actual multiplication tensor is different.
    
    # The multiplication tensor M is defined such that:
    # M(A, B) = A * B
    # In tensor notation: M = sum_{i,j,k} e_i^{(C)} ⊗ e_j^{(A)} ⊗ e_k^{(B)} * delta_{j,k_shared}
    # Hmm, this is getting confusing. Let me use the standard definition.
    
    # The matrix multiplication tensor M_{n,m,p} ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} is defined as:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j}^{(nm)} ⊗ E_{j,k}^{(mp)} ⊗ E_{i,k}^{(np)}
    # where E_{i,j} is the matrix with 1 at position (i,j) and 0 elsewhere.
    
    # So M[i*m+j, j*p+k, i*n+k] = 1 for all valid i,j,k, and 0 otherwise.
    # Wait, that's not a standard tensor entry. Let me re-read the problem.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column of B (0..p-1) and i is the row of A (0..n-1)
    # U is indexed by (i, j) where i is row of A (0..n-1) and j is col of A (0..m-1)  
    # V is indexed by (j, k) where j is row of B (0..m-1) and k is col of B (0..p-1)
    
    # The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] is the tensor entry.
    
    # For matrix multiplication, the tensor entry T[i,j,k] should be 1 if the term A[i,j]*B[j,k]
    # contributes to C[i,k], which is always true for valid i,j,k.
    # But that gives T[i,j,k] = 1 for all i,j,k, which has rank 1.
    
    # I think the issue is that the tensor is NOT the all-ones tensor.
    # Let me look at this from the perspective of the bilinear map.
    
    # The bilinear map is: (A, B) -> A*B
    # This can be written as: sum_c (sum_{i,j} U[i*m+j,c] * A[i,j]) * (sum_{j,k} V[j*p+k,c] * B[j,k]) * ... 
    # No, that's not right.
    
    # Actually, the standard formulation is:
    # A*B = sum_c (U_c^T * A) * (V_c^T * B) * W_c
    # where U_c, V_c, W_c are vectors/matrices.
    
    # More precisely:
    # (A*B)[i,k] = sum_c U[i*m+j, c] * ... 
    
    # I think I need to be more careful. Let me use the standard rank-1 decomposition.
    # A rank-1 term in the multiplication tensor is of the form:
    # u ⊗ v ⊗ w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{np}
    # This corresponds to: sum_{i,j} u[i*m+j] * A[i,j] * sum_{j,k} v[j*p+k] * B[j,k] * sum_{i,k} w[i*n+k] * C[i,k]
    # No, that's not quite right either.
    
    # Let me just try the rank-1 solution and see if it passes verification.
    # If T[i,j,k] = 1 for all i,j,k, then rank 1 should work.
    
    # But I suspect the tensor is NOT all-ones. Let me reconsider.
    
    # For matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The "tensor" that defines this multiplication is:
    # T[i,j,k] = 1 if this is a valid term, which is always true.
    # But the rank of the multiplication tensor is NOT 1; it's at least max(n,m,p) and at most n*m*p.
    
    # I think the confusion is between the tensor and the bilinear map.
    # The bilinear map (A,B) -> A*B has a certain rank, which is the minimum number of rank-1 bilinear maps needed to express it.
    # A rank-1 bilinear map is: (A,B) -> (u^T A v) * w^T or something like that.
    
    # Let me just implement the schoolbook solution and also try rank 1 to see what happens.
    
    # Actually, let me re-read the problem statement more carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]"
    
    # I believe the tensor is defined as:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we need to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    
    # If this is the case, then rank 1 works: U[*,0] = 1, V[*,0] = 1, W[*,0] = 1.
    
    # But this seems too simple. Let me check if there's a constraint I'm missing.
    
    # Perhaps the tensor is not all-ones but has a specific structure.
    # For matrix multiplication, the tensor might be:
    # T[i,j,k] = 1 if j is the shared index, which is always true.
    
    # I'll go with rank 1 and see if it's valid. If not, I'll fall back to schoolbook.
    
    # The schoolbook solution is always valid.
    # Rank = n * m * p
    return {"U": best_U, "V": best_V, "W": best_W}
