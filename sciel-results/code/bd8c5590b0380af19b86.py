import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Construct the multiplication tensor T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # The tensor is of size n x m x p.
    # T[i, j, k] = 1 for all i, j, k because in standard matrix multiplication,
    # C[i, k] = sum_j A[i, j] * B[j, k].
    # The term A[i, j] * B[j, k] appears in C[i, k] with coefficient 1.
    # So the tensor has 1s everywhere.
    
    # 2. Flatten the tensor to compute the lower bound.
    # The "Landsberg" bound (or simply the rank of the flattening) is obtained by
    # flattening the tensor into a matrix.
    # One common flattening is along the first two indices (input A) vs the third (output C part) 
    # or more specifically for matrix multiplication C = A*B:
    # We can view the tensor as a map from V_A* x V_B* to V_C.
    # The rank of the tensor is at least the rank of any flattening.
    # Let's flatten along the output index k (size p) and the other indices (i, j) which form size n*m.
    # Matrix M of size (n*m) x p where M[i*m+j, k] = T[i, j, k] = 1.
    # The rank of this all-ones matrix is 1. This gives a lower bound of 1, which is trivial.
    
    # A better flattening for matrix multiplication:
    # Consider the tensor T in V_i^* \otimes V_j^* \otimes V_k.
    # We want to bound the rank.
    # Let's try flattening (i, k) vs j.
    # Matrix M of size (n*p) x m where M[i*p+k, j] = T[i, j, k] = 1.
    # Rank is 1.
    
    # Let's try flattening (j, k) vs i.
    # Matrix M of size (m*p) x n where M[j*p+k, i] = T[i, j, k] = 1.
    # Rank is 1.
    
    # Wait, the tensor for matrix multiplication is NOT all 1s in a way that makes the flattening rank 1?
    # Let's re-verify.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Yes, it is 1 for all valid indices.
    # So the tensor is the "all ones" tensor of shape (n, m, p).
    # The rank of the all-ones tensor of shape (n, m, p) is known to be min(n, m, p) if we consider standard tensor rank?
    # No, the rank of the all-ones tensor is min(n, m, p) is for the tensor product of vectors?
    # Let's check the rank of the all-ones tensor T of shape (n, m, p).
    # T = 1 \otimes 1 \otimes 1.
    # This is a rank-1 tensor!
    # If the tensor is rank 1, the decomposition is trivial:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    # Then sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1*1*1 = 1.
    # This matches T[i,j,k] = 1.
    
    # BUT, this is not the matrix multiplication tensor!
    # Matrix multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # The tensor T is defined such that C = \sum_t (U_t \cdot A) (B \cdot V_t) ... no.
    # The standard decomposition is:
    # C_{i,k} = \sum_{t=1}^R U_{i,j,t} V_{j,k,t} W_{i,k,t} ? No.
    # The factor matrices are usually defined as:
    # U is n*m x R, V is m*p x R, W is n*p x R.
    # C_{i,k} = \sum_t \sum_j U_{i*m+j, t} V_{j*p+k, t} W_{i*n+k, t} ?
    # No, the standard definition in these tasks is:
    # C_{i,k} = \sum_t ( \sum_j U_{i*m+j, t} A_{i,j} ) ( \sum_l V_{l*p+k, t} B_{l,k} ) ? No.
    
    # Let's look at the seed code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term A_{i,j} * B_{j,k} contributing to C_{i,k} (note W index is k*n+i, which is C_{k,i}? or C_{i,k}?)
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually C is n x p. Index i is row (0..n-1), k is col (0..p-1).
    # Standard linear index for C[i,k] is i*p + k.
    # But the task says W[k*n+i]. This implies C is stored as p x n? Or the index mapping is different.
    # If W[k*n+i] corresponds to the output element, and the loop is i in range(n), k in range(p),
    # then the output element is at index k*n + i.
    # This means the output matrix is effectively transposed or the indexing is (col, row).
    # Regardless, the structure is:
    # One term per (i, j, k) triple.
    # U picks A[i,j], V picks B[j,k], W picks Output[i,k] (with specific indexing).
    # The coefficient is 1.
    
    # So the tensor T[i,j,k] = 1.
    # Is the rank of the all-ones tensor 1?
    # Yes, T[i,j,k] = 1 * 1 * 1.
    # So the rank is 1.
    # If the rank is 1, we can decompose it with R=1.
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    
    # Let's verify if this works.
    # C[i,k] should be sum_j A[i,j] B[j,k].
    # Using the decomposition:
    # The contribution to the output slot corresponding to (i,k) is:
    # sum_t W[index(i,k), t] * (something with U and V)?
    # The standard bilinear form definition for matrix multiplication rank is:
    # C_{i,k} = \sum_{t=1}^R \sum_{j} U_{i,j,t} V_{j,k,t} W_{i,k,t} ?
    # No, the standard definition is:
    # C = \sum_{t=1}^R (A \cdot U_t) \cdot (V_t \cdot B) ?
    # Let's check the seed code's implicit formula.
    # Seed:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    # This sets up a term for each (i,j,k).
    # The formula must be:
    # Output[index(i,k)] += U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t] ?
    # No, that would be cubic in the matrices.
    
    # The correct bilinear form is:
    # C_{i,k} = \sum_t ( \sum_j U_{i*m+j, t} A_{i,j} ) ( \sum_l V_{l*p+k, t} B_{l,k} ) W_{k*n+i, t} ?
    # No, that's not right either.
    
    # Let's look at the standard definition of tensor rank for matrix multiplication.
    # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # The rank is the smallest R such that:
    # M_{i,j,k} = \sum_{t=1}^R U_{i,j,t} V_{j,k,t} W_{i,k,t}
    # Wait, the indices on U, V, W must match the free indices of the tensor.
    # Tensor indices: i (from A), j (shared), k (from B).
    # Output index: (i, k).
    # So the decomposition is:
    # 1 = \sum_{t=1}^R U_{i,j,t} V_{j,k,t} W_{i,k,t}
    # Here U has indices i, j. V has indices j, k. W has indices i, k.
    # In the seed code:
    # U is shape (n*m, R). Index i*m+j corresponds to (i,j).
    # V is shape (m*p, R). Index j*p+k corresponds to (j,k).
    # W is shape (p*n, R). Index k*n+i corresponds to (k,i) which is (i,k) transposed.
    
    # So the equation is:
    # 1 = \sum_{t=1}^R U_{i*m+j, t} V_{j*p+k, t} W_{k*n+i, t}
    
    # If we set R=1, U=1, V=1, W=1:
    # Sum is 1*1*1 = 1.
    # This works!
    
    # So the rank is 1?
    # Why did the seed code use rank 60?
    # Because the seed code was a "schoolbook" placeholder that didn't optimize.
    # And the "Strassen" inspiration reduced it to 54, implying they thought the rank was higher.
    # But if the tensor is the all-ones tensor, its rank is 1.
    
    # Is the matrix multiplication tensor the all-ones tensor?
    # Let's check.
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # The coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So yes, the tensor of coefficients is all 1s.
    
    # Therefore, the rank is 1.
    
    # Let's double check if there are any constraints I missed.
    # "integer/half-integer entries"
    # 1 is an integer.
    # "exactly decompose"
    # Yes.
    
    # Why is this not the known answer?
    # Maybe the "rank" in the context of matrix multiplication usually refers to the rank of the bilinear map, which is indeed the tensor rank.
    # And the tensor rank of the all-ones tensor is 1.
    # However, matrix multiplication is not just "sum of products". It is a specific bilinear map.
    # The bilinear map is (A, B) -> C.
    # The tensor T represents the map.
    # If T = 1 \otimes 1 \otimes 1, then the map is (A, B) -> (sum A)(sum B) ?
    # No.
    # The tensor T_{i,j,k} defines the map as:
    # C_{i,k} = \sum_{j} A_{i,j} B_{j,k}
    # This can be written as:
    # C_{i,k} = \sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
    # If T_{i,j,k} = 1, then C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # This is exactly matrix multiplication.
    
    # So the tensor IS the all-ones tensor.
    # And the rank of the all-ones tensor is 1.
    # So the answer should be 1.
    
    # Let's verify with a small example.
    # n=1, m=1, p=1.
    # A=[a], B=[b], C=[ab].
    # T[0,0,0] = 1.
    # Rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # C[0,0] = 1 * a * b * 1 = ab. Correct.
    
    # n=1, m=2, p=1.
    # A=[a1, a2], B=[b1; b2], C=[a1b1 + a2b2].
    # T[0,0,0] = 1 (for a1b1)
    # T[0,1,0] = 1 (for a2b2)
    # Tensor is 1 x 2 x 1.
    # Flattened (i,k) vs j:
    # i=0, k=0. j=0, 1.
    # Matrix [[1, 1]]. Rank 1.
    # Decomposition R=1:
    # U[0*2+0, 0] = 1, U[0*2+1, 0] = 1
    # V[0*1+0, 0] = 1, V[1*1+0, 0] = 1
    # W[0*1+0, 0] = 1
    # C[0,0] = sum_j U[0*2+j, 0] A[0,j] V[j*1+0, 0] B[j,0] W[0*1+0, 0]
    # = 1*a1*1*b1*1 + 1*a2*1*b2*1 = a1b1 + a2b2. Correct.
    
    # So the rank is indeed 1.
    
    # Why did the previous attempts get 60 or 54?
    # They probably assumed the rank was the number of terms in the schoolbook algorithm (n*m*p) or used Strassen which is for specific structures but still higher than 1.
    # But for the *tensor rank* of the multiplication tensor, it is 1.
    
    # Wait, is it possible that the "rank" requested is not the tensor rank?
    # "Find a rank-minimal exact tensor decomposition"
    # "score is the rank"
    # Yes, it is the tensor rank.
    
    # So the optimal rank is 1.
    
    # Let's construct the rank 1 solution.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
