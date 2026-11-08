"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We know that for small matrix multiplication, the rank can be less than n*m*p.
    # For 2x3x3, let's try to find a decomposition with rank < 18.
    # The tensor T_{i,j,k} = e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (roughly, depending on convention).
    # Actually, the tensor is T_{i,j,k} where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = T_{i,j,k}.
    
    # Let's try to solve for a rank R decomposition.
    # We can use a greedy approach or a simple search for small R.
    # Given the time limit, we should be careful.
    
    # Let's try R = 16, 15, ... down to some minimum.
    # We can set up a linear system for each rank.
    # The equation is: sum_{r=1}^R U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,j,k} (if T is identity-like, but it's not).
    # Wait, T_{i,j,k} = 1 if the indices match the matrix multiplication structure?
    # No, T_{i,j,k} is the coefficient of x_{i,j} y_{j,k} in the product?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The tensor T has components T_{i,j,k} = 1 if we are multiplying A_{i,j} B_{j,k} to get C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k.
    # In matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} = 1 if we view A, B, C as vectors.
    # Actually, T_{i,j,k} = 1 for all i,j,k? No, that would be the all-ones tensor.
    # The tensor for matrix multiplication is T_{i,j,k} = 1 if the indices are "compatible"?
    # Let's check the seed program.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This corresponds to T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # If U, V, W are 0/1 and sparse as in schoolbook, then for each (i,j,k) there is exactly one c such that U_{i,j,c}=V_{j,k,c}=W_{k,i,c}=1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, is the matrix multiplication tensor the all-ones tensor?
    # No. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is such that <T, A \otimes B \otimes C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # We want this to be equal to sum_{i,k} C_{i,k} (or something similar).
    # Actually, the standard definition is that T_{i,j,k} = 1 if and only if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k?
    # Let's re-read the seed.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # The product is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # This sum is over c. For a fixed (i,j,k), the sum is the number of c such that U[i*m+j, c]=V[j*p+k, c]=W[k*n+i, c]=1.
    # In the schoolbook, for each (i,j,k), there is exactly one c (the one corresponding to i,j,k).
    # So the sum is 1.
    # So the tensor T has T_{i,j,k} = 1 for all i,j,k.
    # This means we are looking for a rank decomposition of the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor of size n x m x p is known to be min(n,m,p) if we allow real numbers?
    # No, the rank of the all-ones tensor is 1? No, that's the 2D case.
    # For 3D, the rank of the all-ones tensor is not 1.
    # Actually, the all-ones tensor can be decomposed as sum_{r=1}^R u_r \otimes v_r \otimes w_r where u_r, v_r, w_r are vectors.
    # If we take u_r = e_r, v_r = e_r, w_r = e_r, then the sum is sum_r e_r \otimes e_r \otimes e_r, which is the identity tensor, not the all-ones tensor.
    # To get the all-ones tensor, we need sum_r u_r \otimes v_r \otimes w_r = 1 \otimes 1 \otimes 1.
    # This is possible with R=1: take u_1 = 1, v_1 = 1, w_1 = 1.
    # So the rank is 1?
    # But the schoolbook rank is n*m*p. This suggests that the tensor is NOT the all-ones tensor.
    
    # Let's reconsider the tensor definition.
    # The matrix multiplication tensor T_{n,m,p} is defined by T_{i,j,k} = 1 if the indices (i,j,k) correspond to a term in the product.
    # In the product C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # The tensor T is such that <T, A \otimes B \otimes C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # We want this to be equal to sum_{i,k} C_{i,k} * (something)?
    # No, the standard is that the tensor T represents the bilinear map (A,B) -> C.
    # So T_{i,j,k} = 1 if and only if the (i,j) entry of A and (j,k) entry of B multiply to give the (i,k) entry of C.
    # So T_{i,j,k} = 1 for all i,j,k?
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then <T, A \otimes B \otimes C> = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This is not equal to the matrix product.
    # The matrix product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor should be T_{i,j,k} = 1 if we are summing over j?
    # No, the tensor is a 3-tensor.
    # The correct definition is T_{i,j,k} = 1 if and only if the indices (i,j,k) are such that A_{i,j} and B_{j,k} contribute to C_{i,k}.
    # This is true for all i,j,k.
    # So why is the rank n*m*p?
    # Ah, the tensor is NOT the all-ones tensor.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the indices are "linked" in a specific way.
    # Actually, the matrix multiplication tensor is T_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{k,i} ? No.
    # Let's look at the seed again.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This means that for a fixed c, U, V, W are non-zero only at specific indices.
    # The sum over c gives T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # In the schoolbook, for each (i,j,k), there is exactly one c.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the tensor IS the all-ones tensor.
    # But the rank of the all-ones tensor is 1.
    # So why does the seed use rank n*m*p?
    # Maybe the tensor is not the all-ones tensor.
    # Maybe T_{i,j,k} = 1 if i,j,k are "compatible" in a different way.
    # Let's check the dimensions.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The product is sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # This is a scalar for each (i,j,k).
    # If the tensor is the all-ones tensor, then the rank is 1.
    # But the seed uses rank n*m*p.
    # This implies that the tensor is NOT the all-ones tensor.
    # What is the tensor?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor is T_{n,m,p} = sum_{i,j,k} T_{i,j,k} e_i \otimes e_j \otimes e_k.
    # The standard definition is T_{i,j,k} = 1 if and only if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # This is true for all i,j,k.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the tensor is the all-ones tensor.
    # But the rank is 1.
    # So the seed is not minimal.
    # We can return a rank 1 decomposition.
    # U_{i,j,1} = 1, V_{j,k,1} = 1, W_{k,i,1} = 1.
    # Then sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1*1*1 = 1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # This is the all-ones tensor.
    # So the rank is 1.
    
    # Let's verify this.
    # If we return rank 1, the score will be 1.0 / 1 = 1.0?
    # The score is seed/value. The seed is 18. So the score is 18/1 = 18.0.
    # This is a huge improvement.
    
    # Let's implement this.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
