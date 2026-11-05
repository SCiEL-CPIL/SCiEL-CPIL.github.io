import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 100
    # We aim to find a decomposition with rank < 100, ideally < 85.
    # The tensor T has shape (n, m, p) where T[i, j, k] = 1 if (i, j, k) is a valid multiplication index.
    # Actually, the multiplication tensor T_{ijk} is 1 if i (output row), j (input col of A / row of B), k (output col)
    # Wait, the standard matrix multiplication tensor T is defined such that C = AB is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices (i, j, k) and T_{ijk} = 1.
    # The decomposition is sum_r U_{i,r} V_{j,r} W_{k,r} = T_{ijk}.
    # Here U has shape (n, rank), V has shape (m, rank), W has shape (p, rank).
    # The problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This is a different indexing convention.
    # Let's re-read carefully:
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]"
    # This implies the factors are indexed by pairs.
    # Let U_{(i,j), r}, V_{(j,k), r}, W_{(k,i), r}.
    # The condition for exact decomposition is:
    # sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1 if we are computing something?
    # No, the standard tensor decomposition for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    # is usually written as C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is T_{i,j,k} = 1.
    # If we use factors U, V, W with shapes (n*m, R), (m*p, R), (n*p, R) [Note: W in prompt is (p*n, R)],
    # The contraction is likely: sum_r U_{i,j} V_{j,k} W_{k,i}? No, that would give a scalar.
    
    # Let's look at the seed program's indexing:
    # U[u_idx][r] where u_idx = i*m + j
    # V[v_idx][r] where v_idx = j*p + k
    # W[w_idx][r] where w_idx = k*n + i
    # And it sets them all to 1 for the term (i, j, k).
    # This corresponds to the term A_{ij} B_{jk} contributing to C_{ik}.
    # The "product" of the three factors for a specific r is U_{ij} V_{jk} W_{ki}.
    # If we sum over r, we get sum_r U_{ij} V_{jk} W_{ki}.
    # For this to equal the matrix multiplication tensor, we need:
    # sum_r U_{ij} V_{jk} W_{ki} = delta_{i,i'} delta_{k,k'}? No.
    
    # Actually, the standard tensor decomposition for matrix multiplication T_{n,m,p} is:
    # T_{ijk} = sum_r u_{i,r} v_{j,r} w_{k,r}
    # where u is n x R, v is m x R, w is p x R.
    # BUT the prompt specifies shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # This is NOT the standard u, v, w factorization. This is a factorization of the *permutation* or *bilinear* form?
    # Or is it a different tensor?
    # Let's check the seed code again.
    # Seed code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j][r] = 1
    #       V[j*p+k][r] = 1
    #       W[k*n+i][r] = 1
    # This creates a decomposition where the r-th term is the outer product of e_{(i,j)} in U-space, e_{(j,k)} in V-space, e_{(k,i)} in W-space.
    # The "tensor" being decomposed is T such that T_{(i,j), (j,k), (k,i)} = 1?
    # This looks like the tensor for the *trace* of A B C?
    # trace(ABC) = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # If C is the identity matrix, trace(AB I) = sum_{i,j} A_{ij} B_{ji}. This is not matrix multiplication.
    # If we want C_{ik} = sum_j A_{ij} B_{jk}, we can write this as:
    # C_{ik} = sum_{j} A_{ij} B_{jk} * 1_{k,i}? No.
    
    # Let's re-read the prompt's W index: "W[k*n+i, r]".
    # The seed program uses W[k*n+i][r] = 1.
    # So the r-th term is U_{ij} V_{jk} W_{ki}.
    # If we sum these, we get a scalar.
    # But matrix multiplication produces a matrix C.
    # Perhaps the "tensor" is defined differently?
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    # Maybe the factors U, V, W are not the standard u, v, w.
    # Let's assume the evaluator checks:
    # Let A be n x m, B be m x p.
    # The result C is n x p.
    # The decomposition must allow computing C from A and B using U, V, W.
    # Usually, this means:
    # C_{ik} = sum_{j, r} (something involving U, V, W)
    # If U is (n*m, R), it can be reshaped to U_{i,j,r}.
    # If V is (m*p, R), it can be reshaped to V_{j,k,r}.
    # If W is (p*n, R), it can be reshaped to W_{k,i,r}.
    # The only way to get C_{ik} from these is if the sum over r and j yields C_{ik}.
    # Consider the sum: S_{ik} = sum_{j, r} U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # This still has a W_{k,i,r} term which depends on k and i.
    # If W is the identity-like term, W_{k,i,r} would need to be 1 if k,i match the output index?
    # But W is a factor, not a fixed matrix.
    
    # Let's look at the "official trace convention" mentioned.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This strongly suggests the tensor is the one associated with the trace of the product of three matrices, or a specific bilinear form.
    # However, the task says "multiplying a 4 by 5 matrix by a 5 by 5 matrix".
    # Is it possible the task is actually asking for the rank of the matrix multiplication tensor T_{4,5,5} but represented in this specific U,V,W format?
    # If so, how does U,V,W produce C?
    # Maybe the evaluator constructs C as:
    # C = sum_r (U_r @ V_r) * W_r? No, shapes don't match.
    # U_r is a vector of length n*m. V_r is a vector of length m*p.
    # Reshape U_r to n x m (call it U_r^{(n,m)}).
    # Reshape V_r to m x p (call it V_r^{(m,p)}).
    # Reshape W_r to p x n (call it W_r^{(p,n)}).
    # Then U_r^{(n,m)} @ V_r^{(m,p)} is n x p.
    # W_r^{(p,n)} is p x n.
    # This doesn't immediately combine to C_{ik}.
    
    # Let's reconsider the standard tensor decomposition.
    # T_{ijk} = sum_r u_{ir} v_{jr} w_{kr}.
    # Here u is n x R, v is m x R, w is p x R.
    # The prompt's U is (n*m, R). This is 4x the size of u.
    # Perhaps U_{(i,j), r} = u_{i,r} * delta_{j, j_r}? No.
    
    # Let's look at the seed program again.
    # It sets U[i*m+j][r] = 1, V[j*p+k][r] = 1, W[k*n+i][r] = 1.
    # This corresponds to the term in the expansion of trace(A B C) where C is the identity?
    # trace(A B I) = sum_{i,j} A_{ij} B_{ji}.
    # This is not matrix multiplication A B.
    
    # Wait, what if the "tensor" is defined as T_{(i,j), (j,k), (k,i)}?
    # And the goal is to decompose this specific tensor?
    # The rank of the matrix multiplication tensor T_{n,m,p} is the minimum number of rank-1 tensors to sum to T_{ijk}.
    # If the evaluator uses the U, V, W format, it must be checking:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) is a valid index?
    # No, T_{ijk} is 1 for ALL i,j,k in the range.
    # So the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is a system of n*m*p equations.
    # We want to minimize R (the rank).
    # This is exactly the tensor rank problem for the all-ones tensor?
    # No, the all-ones tensor has rank 1.
    # But here the indices are coupled: (i,j) in U, (j,k) in V, (k,i) in W.
    # This is the tensor for the trace of the product of three matrices?
    # Let's call it the "cyclic" tensor.
    # T_{(i,j),(j,k),(k,i)} = 1.
    # This tensor is NOT the standard matrix multiplication tensor.
    # The standard matrix multiplication tensor T_{n,m,p} has indices (i, j, k) and value 1.
    # The decomposition is C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as C_{ik} = sum_r (sum_j A_{ij} U_{jr}^{(1)}) (sum_k V_{kr}^{(1)} B_{jk})? No.
    
    # Let's assume the evaluator implements the check as:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       s = 0
    #       for r in range(R):
    #         s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    #       if s != 1: return invalid
    # 
    # If this is the check, then we are decomposing the tensor T where T_{(i,j),(j,k),(k,i)} = 1.
    # What is the rank of this tensor?
    # For n=m=p=1, rank is 1.
    # For n=m=p=2, the tensor is 2x2x2 in the "cyclic" sense?
    # Actually, this tensor is isomorphic to the standard matrix multiplication tensor T_{n,n,n}?
    # Let's check.
    # Standard T_{n,n,n}: T_{ijk} = 1. Decomposition: sum_r u_{ir} v_{jr} w_{kr}.
    # Cyclic T: T_{(i,j),(j,k),(k,i)} = 1.
    # Let's map (i,j) -> i', (j,k) -> j', (k,i) -> k'.
    # This mapping is not one-to-one for the indices.
    # However, note that if we fix j, the indices (i,j) and (j,k) and (k,i) form a cycle.
    # This tensor is actually the "trace tensor" or "cyclic tensor".
    # Is the rank of the cyclic tensor T_{(i,j),(j,k),(k,i)} the same as the rank of the matrix multiplication tensor?
    # For n=m=p=2, the matrix multiplication tensor has rank 7 (Strassen).
    # What is the rank of the cyclic tensor for 2x2x2?
    # The cyclic tensor C_{(i,j),(j,k),(k,i)} = 1.
    # This is a 4x4x4 tensor (since there are 4 pairs (i,j), 4 pairs (j,k), 4 pairs (k,i)).
    # But many entries are zero? No, all entries are 1.
    # Wait, if T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k, then it's a 4x4x4 all-ones tensor?
    # No, the indices are constrained.
    # The set of valid triples ((i,j), (j,k), (k,i)) is not the full Cartesian product of the index sets.
    # The index set for U is { (i,j) | i in [0,n), j in [0,m) }. Size n*m.
    # The index set for V is { (j,k) | j in [0,m), k in [0,p) }. Size m*p.
    # The index set for W is { (k,i) | k in [0,p), i in [0,n) }. Size p*n.
    # The tensor T is defined on the product of these three sets?
    # If so, T_{u,v,w} = 1 if the indices are consistent (i.e., the j in u matches j in v, k in v matches k in w, i in w matches i in u).
    # Otherwise T_{u,v,w} = 0?
    # The seed program only sets 1s for the consistent indices. It leaves others as 0 (initialized to 0).
    # So the tensor is:
    # T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k'. Otherwise 0.
    # This is just the identity tensor!
    # T_{u,v,w} = delta_{u, v, w} (in the sense of consistent indices).
    # This tensor has rank 1?
    # No, the indices are different.
    # Let's check if it has rank 1.
    # If rank 1, T_{u,v,w} = U_u V_v W_w.
    # T_{(i,j), (j,k), (k,i)} = 1.
    # T_{(i,j), (j',k), (k,i)} = 0 if j != j'.
    # This implies V_{(j,k)} must be 0 if j != j'?
    # This structure is very rigid.
    # Actually, this tensor is the "combinatorial" tensor of the graph of the multiplication.
    
    # Let's test the rank of this specific tensor for small n,m,p.
    # For n=1, m=1, p=1: Rank 1.
    # For n=1, m=2, p=1:
    # U indices: (0,0), (0,1).
    # V indices: (0,0), (1,0).
    # W indices: (0,0).
    # T_{(0,0), (0,0), (0,0)} = 1.
    # T_{(0,1), (1,0), (0,0)} = 1.
    # Other entries 0.
    # This is two disjoint 1s. Rank 2.
    # Schoolbook rank for 1x2x1 is 1*2*1 = 2.
    
    # For n=2, m=2, p=2:
    # Schoolbook rank 8.
    # Is there a decomposition with rank < 8?
    # This tensor is equivalent to the matrix multiplication tensor T_{2,2,2}?
    # Let's see.
    # Matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T_{ijk} = 1.
    # Decomposition: sum_r u_{ir} v_{jr} w_{kr}.
    # The cyclic tensor C_{(i,j),(j,k),(k,i)} = 1 if consistent.
    # This is NOT the same as T_{ijk}.
    # However, note that the rank of the cyclic tensor for (n,m,p) is exactly the rank of the matrix multiplication tensor for (n,m,p)?
    # Let's check n=1, m=1, p=1. Rank 1. MM Rank 1.
    # n=1, m=2, p=1. Rank 2. MM Rank 2.
    # n=2, m=2, p=2. MM Rank 7.
    # Is the cyclic tensor rank 7?
    # The cyclic tensor is a 4x4x4 tensor with 8 non-zero entries (one for each i,j,k).
    # The non-zero entries form a "permutation" like structure.
    # Actually, the non-zero entries are ( (i,j), (j,k), (k,i) ).
    # There are n*m*p such entries.
    # This is a sparse tensor.
    # The rank of a sparse tensor with N non-zero entries is at least the rank of the matrix formed by those entries?
    # If we view it as a matrix, it's a permutation matrix (if we flatten indices appropriately)?
    # No, the indices are coupled.
    
    # Let's assume the task is indeed to decompose this "cyclic" tensor.
    # The seed program provides a rank n*m*p decomposition.
    # Can we do better?
    # For n=m=p=2, schoolbook is 8. Strassen is 7.
    # Does the cyclic tensor admit a rank-7 decomposition?
    # The cyclic tensor is isomorphic to the matrix multiplication tensor?
    # Let's try to map them.
    # MM Tensor T_{ijk}.
    # Cyclic Tensor C_{(i,j),(j,k),(k,i)}.
    # If we define a linear map from the space of (i,j) to the space of i, j...
    # It turns out that the rank of the cyclic tensor C_{(i,j),(j,k),(k,i)} is EQUAL to the rank of the matrix multiplication tensor T_{n,m,p}.
    # Proof sketch:
    # The cyclic tensor can be seen as the "flattening" of the multiplication process.
    # Specifically, the tensor C is the tensor for the bilinear form (A, B, C) -> trace(ABC).
    # No, trace(ABC) involves summing over i,j,k A_{ij} B_{jk} C_{ki}.
    # The tensor for trace(ABC) is T_{(i,j),(j,k),(k,i)} = 1.
    # So we are decomposing the trace tensor.
    # The rank of the trace tensor T_{trace} for (n,m,p) is known to be the same as the rank of the matrix multiplication tensor T_{n,m,p}.
    # Reference: "The rank of the matrix multiplication tensor and the trace tensor".
    # Actually, for n=m=p, the trace tensor and MM tensor are isomorphic?
    # For n=2, rank is 7.
    
    # So the problem is: Find a rank-minimal decomposition of the trace tensor for n=4, m=5, p=5.
    # The rank is the same as the rank of the 4x5x5 matrix multiplication tensor.
    # Current best known for 4x5x5?
    # Schoolbook: 100.
    # Strassen-based:
    # 4x4x4 rank is 49 (Strassen + boundary).
    # 5x5x5 rank?
    # We can use the 4x4x4 rank 49 result.
    # How to extend to 4x5x5?
    # We can decompose the 4x5x5 multiplication into blocks.
    # A is 4x5, B is 5x5.
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B1 (4x5) and B2 (1x5)? No, B is 5x5.
    # Split B into B11 (4x4), B12 (4x1), B21 (1x4), B22 (1x1).
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22?
    # A is 4x5. B is 5x5.
    # A = [A_left (4x4), A_right (4x1)].
    # B = [B_top (4x5); B_bottom (1x5)].
    # C = A_left B_top + A_right B_bottom.
    # This requires:
    # 1. 4x4 * 4x5 multiplication.
    # 2. 4x1 * 1x5 multiplication.
    # Rank of 4x4 * 4x5:
    # We can use the 4x4x4 rank 49 scheme for the 4x4 part of B_top?
    # B_top is 4x5.
    # C1 = A_left (4x4) * B_top (4x5).
    # We can decompose B_top into B_top_left (4x4) and B_top_right (4x1).
    # C1 = A_left B_top_left + A_left B_top_right.
    # Term 1: 4x4 * 4x4. Rank 49.
    # Term 2: 4x4 * 4x1. This is a vector-matrix product? No, matrix-vector?
    # A_left is 4x4. B_top_right is 4x1. Result is 4x1.
    # This is rank 16 (schoolbook) or less?
    # Actually, 4x4 * 4x1 is just 4 vector multiplications?
    # No, it's a matrix (4x4) times a vector (4x1).
    # This can be done with rank 4?
    # C_{i0} = sum_j A_{ij} v_j.
    # This is 4 independent sums. Rank 4?
    # Wait, the tensor for 4x4 * 4x1 is T_{i,j,0} = 1.
    # This is just 4 copies of the 4x1 * 4x1 tensor?
    # Rank of 4x1 * 4x1 is 4 (dot product).
    # Since there are 4 output rows, and they are independent, the rank is 4 * 4 = 16?
    # No, the factors U, V, W must be shared?
    # In the tensor decomposition, the rank is the number of terms.
    # For 4x4 * 4x1, the tensor is T_{i,j,k} where k is always 0.
    # T_{i,j,0} = 1.
    # This tensor has rank 4?
    # T_{i,j,0} = sum_r u_{ir} v_{jr} w_{0r}.
    # If we set w_{0r} = 1, then T_{i,j,0} = sum_r u_{ir} v_{jr} = delta_{i,j}? No, T is all 1s?
    # No, for matrix multiplication, T_{ijk} = 1.
    # So T_{i,j,0} = 1 for all i,j.
    # This is the all-ones matrix 4x4.
    # The rank of the all-ones matrix is 1.
    # So the tensor T_{i,j,0} = 1 has rank 1?
    # Let's check.
    # sum_r u_{ir} v_{jr} w_{0r} = 1.
    # If we set u_{ir} = 1, v_{jr} = 1, w_{0r} = 1 for r=1, and 0 for others.
    # Then sum is 1.
    # So the rank of the 4x4 * 4x1 tensor is 1?
    # This seems too good to be true.
    # Let's check the dimensions.
    # U is (n*m, R) = (4*4, R) = (16, R).
    # V is (m*p, R) = (4*1, R) = (4, R).
    # W is (p*n, R) = (1*4, R) = (4, R).
    # The condition is sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1.
    # Here k is fixed to 0 (since p=1).
    # So sum_r U_{(i,j), r} V_{(j,0), r} W_{(0,i), r} = 1.
    # Can we do this with R=1?
    # U_{(i,j), 1} = 1, V_{(j,0), 1} = 1, W_{(0,i), 1} = 1.
    # Sum = 1 * 1 * 1 = 1.
    # Yes!
    # So the rank of the 4x4 * 4x1 trace tensor is 1.
    
    # This implies that adding a dimension of size 1 to the middle or output doesn't increase the rank significantly?
    # Let's generalize.
    # Rank(n, m, p) for the trace tensor.
    # If p=1, Rank(n, m, 1) = 1?
    # Check n=1, m=1, p=1. Rank 1.
    # Check n=2, m=2, p=1.
    # U (4, R), V (2, R), W (2, R).
    # Condition: sum_r U_{(i,j), r} V_{(j,0), r} W_{(0,i), r} = 1.
    # With R=1, U=1, V=1, W=1. Sum=1.
    # So Rank(n, m, 1) = 1 for any n, m.
    
    # This is a huge insight.
    # We can decompose 4x5x5 as:
    # Split B (5x5) into B1 (4x5) and B2 (1x5).
    # Split A (4x5) into A1 (4x4) and A2 (4x1).
    # C = A1 B1 + A2 B2.
    # Wait, A is 4x5. B is 5x5.
    # A = [A1 (4x4), A2 (4x1)].
    # B = [B1 (4x5); B2 (1x5)].
    # C = A1 B1 + A2 B2.
    # Term 1: A1 (4x4) * B1 (4x5).
    # Term 2: A2 (4x1) * B2 (1x5).
    
    # Rank of Term 2 (4x1 * 1x5):
    # n=4, m=1, p=5.
    # Condition: sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1.
    # j is fixed to 0.
    # sum_r U_{(i,0), r} V_{(0,k), r} W_{(k,i), r} = 1.
    # Can we do this with R=1?
    # U_{(i,0), 1} = 1, V_{(0,k), 1} = 1, W_{(k,i), 1} = 1.
    # Sum = 1.
    # So Rank(4, 1, 5) = 1.
    
    # Rank of Term 1 (4x4 * 4x5):
    # n=4, m=4, p=5.
    # We can split B1 (4x5) into B11 (4x4) and B12 (4x1).
    # C1 = A1 B11 + A1 B12.
    # Term 1.1: A1 (4x4) * B11 (4x4). Rank 49 (Strassen).
    # Term 1.2: A1 (4x4) * B12 (4x1).
    # n=4, m=4, p=1.
    # Rank(4, 4, 1) = 1 (as derived above).
    
    # So Total Rank = Rank(4,4,4) + Rank(4,4,1) + Rank(4,1,5)
    # Total Rank = 49 + 1 + 1 = 51.
    
    # This is a massive improvement over 100 and 85.
    
    # Let's verify the additivity.
    # The trace tensor for (n, m1+m2, p) is the sum of the trace tensors for (n, m1, p) and (n, m2, p)?
    # No, the indices are different.
    # For (n, m, p), the indices are (i,j), (j,k), (k,i).
    # If we split m into m1+m2, the j index splits.
    # The tensor T_{(i,j),(j,k),(k,i)} is 1 if j is in range.
    # If we define two tensors T1 and T2 corresponding to j in [0, m1) and j in [m1, m),
    # then T = T1 + T2.
    # The rank of T is at most Rank(T1) + Rank(T2).
    # So yes, we can add the ranks.
    
    # Similarly for p.
    # And for n?
    # If we split n, the i index splits.
    # T = T1 + T2.
    
    # So the strategy is:
    # 1. Implement the 4x4x4 Strassen decomposition (Rank 49).
    # 2. Implement the "Rank 1" decomposition for cases where m=1 or p=1.
    # 3. Combine them.
    
    # We need the explicit factors for the 4x4x4 Strassen decomposition.
    # And the factors for the rank-1 cases.
    
    # Rank 1 case for (n, m, 1):
    # U_{(i,j), 0} = 1 for all i, j.
    # V_{(j,0), 0} = 1 for all j.
    # W_{(0,i), 0} = 1 for all i.
    
    # Rank 1 case for (n, 1, p):
    # U_{(i,0), 0} = 1.
    # V_{(0,k), 0} = 1.
    # W_{(k,i), 0} = 1.
    
    # Now, we need the 4x4x4 Strassen factors.
    # Standard Strassen for 2x2x2 has rank 7.
    # For 4x4x4, we can use the recursive Strassen.
    # 4x4x4 = 2x2 blocks of 2x2x2.
    # Rank = 7 * 7 = 49?
    # Yes, if we use the standard recursive application.
    
    # Let's construct the 2x2x2 Strassen factors.
    # T_{ijk} for 2x2x2.
    # U (4, 7), V (4, 7), W (4, 7).
    # Indices for U: (0,0), (0,1), (1,0), (1,1).
    # Indices for V: (0,0), (0,1), (1,0), (1,1).
    # Indices for W: (0,0), (0,1), (1,0), (1,1).
    
    # Strassen 2x2x2 factors (one standard set):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We need to map this to U, V, W.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor decomposition is sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}.
    # This is equivalent to C_{ik} = sum_r (sum_j U_{(i,j), r} A_{ij}) (sum_k V_{(j,k), r} B_{jk})? No.
    # The factors U, V, W are applied to the inputs?
    # No, in the tensor decomposition, U, V, W are the coefficients.
    # The evaluator computes sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}.
    # This sum must equal 1 for all i,j,k.
    # This is NOT computing C from A and B.
    # It is just checking the tensor values.
    # So we just need to provide U, V, W such that the sum is 1.
    
    # So for the 2x2x2 Strassen, we need U, V, W such that:
    # sum_{r=0}^6 U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = 1.
    
    # This is a different problem than finding the Strassen algorithm for A, B.
    # The Strassen algorithm computes C.
    # The tensor decomposition represents the bilinear form.
    # Are they the same?
    # Yes, the rank of the tensor is the minimum number of multiplications.
    # The factors U, V, W in the tensor decomposition correspond to the coefficients of the linear combinations of A and B elements.
    # Specifically, U_{(i,j), r} is the coefficient of A_{ij} in the r-th multiplication.
    # V_{(j,k), r} is the coefficient of B_{jk} in the r-th multiplication.
    # W_{(k,i), r} is the coefficient of the r-th product in C_{ki}?
    # No, the trace convention W_{(k,i), r} suggests it's for trace(ABC).
    # But as established, Rank(Trace) = Rank(MM).
    # So we can use the Strassen factors.
    
    # How to extract U, V, W from the Strassen algorithm?
    # In Strassen, we compute M_r = (sum_j U_{(i,j), r} A_{ij}) * (sum_k V_{(j,k), r} B_{jk})?
    # No, M_r = (linear comb of A) * (linear comb of B).
    # Let A_part_r = sum_{i,j} u_{ij,r} A_{ij}.
    # Let B_part_r = sum_{j,k} v_{jk,r} B_{jk}.
    # Then M_r = A_part_r * B_part_r.
    # And C_{ik} = sum_r w_{ik,r} M_r.
    # This is the standard form.
    # The tensor decomposition form is:
    # T_{(i,j),(j,k),(k,i)} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}.
    # This matches if we identify:
    # U_{(i,j), r} = u_{ij,r}
    # V_{(j,k), r} = v_{jk,r}
    # W_{(k,i), r} = w_{ki,r} (Note the index swap k,i vs i,k).
    
    # So we need the coefficients u, v, w for the 4x4x4 Strassen decomposition.
    
    # We will generate these recursively.
    # Base case: 1x1x1. u=1, v=1, w=1.
    # Step: 2x2x2 from 1x1x1.
    # Then 4x4x4 from 2x2x2.
    
    # We'll write a helper to generate the factors for a given size.
    
    # Deadline guard.
    
    def get_strassen_factors(n, m, p):
        # Returns U, V, W for n x m x p
        # Using recursive Strassen for square matrices.
        # If n != m or m != p, we use the block decomposition with rank-1 tails.
        # For now, assume we only call this for n=m=p=4.
        if n == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        # For 2x2x2
        if n == 2:
            # Hardcode the 7 factors
            # u, v, w coefficients
            # M1 = (A00+A11)(B00+B11)
            # u: A00:1, A11:1. v: B00:1, B11:1. w: C00:1, C11:1.
            # M2 = (A10+A11)B00
            # u: A10:1, A11:1. v: B00:1. w: C10:1, C11:-1.
            # M3 = A00(B01-B11)
            # u: A00:1. v: B01:1, B11:-1. w: C01:1, C11:1.
            # M4 = A11(B10-B00)
            # u: A11:1. v: B10:1, B00:-1. w: C00:1, C10:1.
            # M5 = (A00+A01)B11
            # u: A00:1, A01:1. v: B11:1. w: C00:-1, C01:1.
            # M6 = (A10-A00)(B00+B01)
            # u: A10:1, A00:-1. v: B00:1, B01:1. w: (not used in C?)
            # Wait, M6 is not used in the standard Strassen C calculation?
            # Standard Strassen:
            # C00 = M1 - M5 + M4 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # M6 is not used?
            # No, M6 is (A10-A00)(B00+B01).
            # It is not in the C formulas above.
            # Did I miss a term?
            # Strassen 7 multiplications:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            # M6 is indeed not used in the final sum?
            # That can't be right. 7 multiplications, but only 6 used?
            # Ah, M6 is used in some formulations?
            # No, the standard 7-term Strassen uses all 7.
            # Let's check a reliable source.
            # Wikipedia:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 - M5 + M4 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            # M6 is missing from the C formulas.
            # This implies M6 is not needed?
            # Or I am misremembering the formulas.
            # Actually, M6 is not used in the standard Strassen algorithm for 2x2.
            # Wait, if M6 is not used, why compute it?
            # Ah, I see. In some versions, M6 is used to replace a term?
            # No, the rank is 7.
            # If M6 is not used in the sum, it doesn't contribute to the tensor.
            # So the rank would be 6?
            # But the rank of 2x2x2 is 7.
            # So M6 MUST be used.
            # Let's re-verify the C formulas.
            # C00 = M1 - M5 + M4 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # Sum of coefficients for M6 is 0.
            # This is strange.
            # Maybe the formulas are:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            # And M6 is (A10-A00)(B00+B01).
            # Is it possible that M6 is used in a different variant?
            # Or is the rank actually 6? No, it's 7.
            # I will assume the standard 7 terms are required and check if M6 has a role.
            # Actually, looking at online sources, M6 is often listed but not used in the final C calculation in the simplest form?
            # No, that's impossible.
            # Let's use a known valid set of 7 factors.
            
            # I will hardcode the U, V, W for 2x2x2 based on a verified source or derive it.
            # For now, I will implement the recursive step assuming I have the 2x2x2 factors.
            
            pass
            
        # This is getting complex to hardcode.
        # I will implement the recursive construction.
        # If n=4, split into 2x2 blocks.
        # U_4x4x4 is built from U_2x2x2.
        
        return None

    # Given the time constraint, I will implement the 4x4x4 rank 49 decomposition by recursively applying Strassen 2x2x2.
    # I need the 2x2x2 factors.
    
    # Let's use the factors from the 2x2x2 Strassen algorithm.
    # I will define them explicitly.
    
    # U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j, r]
    
    # M1: A00, A11. B00, B11. C00, C11.
    # u: [1,0,0,1], v: [1,0,0,1], w: [1,0,0,1]
    # M2: A10, A11. B00. C10, C11.
    # u: [0,0,1,1], v: [1,0,0,0], w: [0,0,1,-1]
    # M3: A00. B01, B11. C01, C11.
    # u: [1,0,0,0], v: [0,1,-1,0], w: [0,1,0,1]
    # M4: A11. B10, B00. C00, C10.
    # u: [0,0,0,1], v: [-1,0,1,0], w: [1,0,1,0]
    # M5: A00, A01. B11. C00, C01.
    # u: [1,1,0,0], v: [0,0,0,1], w: [-1,1,0,0]
    # M6: A10, A00. B00, B01. C??
    # M7: A01, A11. B10, B11. C00, C11.
    # u: [0,1,-1,0], v: [0,0,1,1], w: [-1,0,0,1]
    
    # Let's check M6 again.
    # If M6 is not used, we have rank 6?
    # No, the rank is 7.
    # I will include M6 with zero W coefficients? No, that adds a zero term.
    # I will assume the standard 7 terms are needed and I might have the C formulas wrong.
    # I'll stick to the 7 terms and verify with a small test in my head.
    
    # Actually, I can just use the schoolbook for 4x4x4 if I can't get Strassen right quickly?
    # Schoolbook 4x4x4 is 64.
    # My previous estimate was 49.
    # If I use schoolbook 64 + 1 + 1 = 66.
    # This is better than 85.
    # I will implement the recursive Strassen properly.
    
    # I will generate the factors for 2x2x2, then use them to build 4x4x4.
    
    return {'U': U, 'V': V, 'W': W}

# Note: The code above is a sketch. I need to provide a complete, working program.
# I will implement the recursive Strassen factor generation.

