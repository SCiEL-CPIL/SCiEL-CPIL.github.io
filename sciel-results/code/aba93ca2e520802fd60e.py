"""Matrix multiplication 2x2 * 2x13 - tensor decomposition search."""
import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Target: minimize rank of tensor decomposition for (n,m) x (m,p) -> (n,p)
    # Standard rank is n*m*p = 52
    # We know rank(2x2 * 2x2) = 7 (Strassen)
    # For 2x2 * 2x13, we can use block structure
    
    # The multiplication tensor T has indices (i,j,k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Output convention: W[k*n+i, r]
    
    # Strategy: Use Strassen-like decomposition for the 2x2 part
    # A (2x2) * B (2x13) = C (2x13)
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Strassen for 2x2 * 2x2 uses 7 multiplications:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 2x2 * 2x13, each "element" of B is a vector of length 13.
    # So M1 = (A11+A22) * (B11+B22) where B11, B22 are vectors of length 13
    # This gives a vector of length 13 for each multiplication.
    
    # The rank would be 7 * 13 = 91? No, that's worse than 52.
    # Wait, the standard decomposition has rank n*m*p = 2*2*13 = 52.
    
    # Actually, for matrix multiplication (n,m) x (m,p) -> (n,p),
    # the trivial rank is n*m*p.
    # For 2x2 * 2x2, trivial is 8, Strassen is 7.
    # For 2x2 * 2x13, trivial is 52.
    
    # Can we do better than 52?
    # The tensor is T_{ijk} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where output is (i', k') and inputs are (i,j) and (j',k)
    # More precisely: C[i', k'] = sum_j A[i', j] * B[j, k']
    # Tensor: T[i', j, j', k'] = 1 if i'=i and j'=j and k'=k
    # In our indexing: U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r]
    # Wait, let me re-read the convention.
    
    # U has shape (n*m, rank) - indices (i, j) for A
    # V has shape (m*p, rank) - indices (j, k) for B  
    # W has shape (p*n, rank) - indices (k, i) for C, with W[k*n+i, r]
    
    # The decomposition says: C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. Let me think again.
    
    # For matrix multiplication A*B = C where A is n x m, B is m x p, C is n x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # The tensor decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's not how it works either.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # T[i,j,k,l] = delta_{i,i'} delta_{j,j'} delta_{k,k'} where we have
    # C[i', k'] = sum_{j'} A[i', j'] B[j', k']
    
    # The factorization is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # and (U_r ⊗ V_r ⊗ W_r)[i*m+j, j'*p+k, k'*n+i'] = U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i']
    
    # We need: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = delta_{j,j} (always 1 when j=j)
    # Wait, I'm confusing myself. Let me use the standard formulation.
    
    # The matrix multiplication tensor T in R^{(n*m) x (m*p) x (n*p)} has entries:
    # T[a, b, c] where a = i*m + j (index into A), b = j'*p + k (index into B), c = k'*n + i' (index into C)
    # T[a, b, c] = 1 if j = j' (the shared index matches), else 0
    # And we need i' = i, k' = k for the output to be correct.
    
    # Actually: T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' AND i'=i AND k'=k, else 0
    # This is a diagonal-like tensor.
    
    # The trivial decomposition: for each (i,j,k), set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # with rank = n*m*p = 52.
    
    # To beat this, we need to find a lower rank decomposition.
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x2 * 2x13, what's the minimal rank?
    
    # Lower bound: The rank is at least max(n*m, m*p, n*p) = max(4, 26, 26) = 26
    # Also at least the slice rank.
    
    # Let's try a Strassen-inspired approach.
    # A is 2x2, B is 2x13.
    # We can think of B as two row-vectors of length 13: B[0,:], B[1,:]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Using Strassen for the 2x2 part:
    # Let's define:
    # S1 = A[0,0] + A[1,1]
    # S2 = A[1,0] + A[1,1]
    # S3 = A[0,0]
    # S4 = A[1,1]
    # S5 = A[0,0] + A[0,1]
    # S6 = A[1,0] - A[0,0]
    # S7 = A[0,1] - A[1,1]
    
    # For the B side, since B has 2 rows of length 13:
    # T1 = B[0,:] + B[1,:]  (vector of length 13)
    # T2 = B[0,:]  (vector of length 13)
    # T3 = B[1,:] - B[0,:]  (vector of length 13)
    # T4 = B[1,:] - B[0,:]  (same as T3)
    # T5 = B[1,:]  (vector of length 13)
    # T6 = B[0,:] + B[1,:]  (same as T1)
    # T7 = B[1,:]  (same as T5)
    
    # Products:
    # P1 = S1 * T1  (scalar * vector = vector of length 13)
    # P2 = S2 * T2  (vector of length 13)
    # P3 = S3 * T3  (vector of length 13)
    # P4 = S4 * T4  (vector of length 13)
    # P5 = S5 * T5  (vector of length 13)
    # P6 = S6 * T6  (vector of length 13)
    # P7 = S7 * T7  (vector of length 13)
    
    # Then:
    # C[0,k] = P1[k] - P4[k] + P3[k] + P6[k]
    # C[1,k] = P2[k] + P4[k] - P5[k] - P7[k]
    
    # Wait, let me verify with standard Strassen:
    # C11 = M1 - M4 + M3 + M6
    # C12 = M7 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M2
    
    # Hmm, let me use the correct Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M3 + M6
    # C12 = M7 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M2
    
    # In our case, B is 2x13, so B11, B12, B21, B22 are vectors of length 13.
    # Actually, B[j, k] for j in {0,1} and k in {0,...,12}.
    # So "B11" in Strassen terminology would be B[0, :] (first row, all 13 columns)
    # "B12" would be... wait, Strassen is for 2x2 * 2x2.
    
    # Let me reconsider. In Strassen for 2x2 * 2x2:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # For 2x2 * 2x13, B is 2x13. We can think of this as 13 separate 2x2 * 2x1 multiplications
    # or as a 2x2 * 2x13 problem.
    
    # The key insight: each column of B is a 2x1 vector, and we're computing A * (each column).
    # But that doesn't directly help with rank reduction.
    
    # Alternative: Use the fact that the 2x2 matrix multiplication has rank 7.
    # For 2x2 * 2x13, we can view it as 13 instances of 2x2 * 2x1, but they share the A matrix.
    
    # Actually, let me think about this differently.
    # The tensor T has dimensions (n*m) x (m*p) x (n*p) = 4 x 26 x 26.
    # The trivial rank is 52.
    # 
    # Can we achieve rank < 52?
    # 
    # One approach: Use the Strassen decomposition for the 2x2 part.
    # The 2x2 matrix multiplication tensor has rank 7.
    # For 2x2 * 2x13, we can write:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # = sum_{i,j} A[i,j] * B[j,k] * delta_{output}
    
    # Using Strassen: the 2x2 multiplication can be done with 7 scalar multiplications.
    # For 2x2 * 2x13, each "scalar multiplication" becomes a scalar-vector multiplication
    # (scalar times a vector of length 13), which still counts as rank 1 per component.
    
    # So the rank would be 7 * 13 = 91? That's worse.
    
    # Hmm, let me reconsider. The issue is that in the tensor decomposition,
    # each rank-1 term contributes to all 13 output positions simultaneously.
    
    # Actually, I think the correct way to think about it:
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # 
    # A rank-1 decomposition term is U_r * V_r * W_r where:
    # U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i']
    # 
    # For the trivial decomposition, each term corresponds to a fixed (i, j, k):
    # U_r is 1 at position i*m+j, V_r is 1 at position j*p+k, W_r is 1 at position k*n+i.
    # This gives exactly T[i*m+j, j*p+k, k*n+i] = 1.
    
    # To reduce rank, we need to find linear combinations.
    # 
    # Let's try a different approach: use the algebraic structure.
    # Consider the map (A, B) -> A*B where A is 2x2 and B is 2x13.
    # We can embed this into a larger algebraic structure.
    
    # Actually, let me just try to use random search / optimization to find a low-rank decomposition.
    # But that might be too slow.
    
    # Let me try a structured approach based on Strassen.
    # 
    # Key idea: For the 2x2 * 2x2 case, Strassen achieves rank 7.
    # For 2x2 * 2x13, we can think of it as computing 13 columns of the product.
    # Each column computation is a 2x2 * 2x1 multiplication.
    # 
    # The 2x2 * 2x1 multiplication: C[i] = sum_j A[i,j] B[j]
    # This has rank 4 (trivial) or can it be done with less?
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is a bilinear map from (2x2) x (2) -> (2).
    # The tensor has dimensions 4 x 2 x 2.
    # Trivial rank: 4*2*2 / ... no, trivial rank is n*m*p = 2*2*2 = 8? No.
    # 
    # Wait, for (n,m) x (m,p) -> (n,p), the trivial rank is n*m*p.
    # For 2x2 * 2x1: n=2, m=2, p=1, trivial rank = 2*2*1 = 4.
    # 
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] for n=2,m=2,p=1:
    # T[i*2+j, j'*1+0, 0*2+i'] = delta_{j,j'} * delta_{i,i'} * delta_{0,0}
    # = delta_{j,j'} * delta_{i,i'}
    # 
    # So T has non-zero entries at (i*2+j, j, i) for i,j in {0,1}.
    # That's 4 non-zero entries. Trivial rank is 4.
    # Can we do better? The slice rank lower bound... 
    # 
    # Actually, for this specific case, the rank is exactly 4 because
    # the four terms are linearly independent in a certain sense.
    
    # OK, let me just go with the trivial decomposition of rank 52 for now,
    # and try to optimize.
    
    # Actually, wait. Let me reconsider the problem.
    # The inspiration mentions score 1.1304 which corresponds to rank 46.
    # So rank 46 is achievable.
    
    # Let me think about how to get rank 46.
    # 52 - 46 = 6 savings.
    
    # One approach: Use Strassen for a 2x2 sub-block.
    # If we use Strassen for one 2x2 * 2x2 sub-problem, we save 1 rank (8 -> 7).
    # But we have 13 columns, so we could potentially save more.
    
    # Actually, here's an idea:
    # Write B as [B_left, B_right] where B_left is 2x6 and B_right is 2x7.
    # No, that doesn't help directly.
    
    # Better idea: Use the Strassen decomposition for the entire 2x2 * 2x13 problem.
    # The 2x2 matrix multiplication has rank 7.
    # For 2x2 * 2x13, we can use a "tensor product" type argument.
    
    # Actually, let me just implement the trivial decomposition and return it.
    # The score will be 1.0 which is the same as the seed.
    
    # But the inspirations show rank 46 is achievable. Let me try to construct that.
    
    # Strategy: Use Strassen for a 2x2 * 2x2 block and trivial for the rest.
    # A is 2x2, B is 2x13.
    # Split B into B1 (2x2) and B2 (2x11).
    # C = A * [B1, B2] = [A*B1, A*B2]
    # A*B1 uses Strassen (rank 7), A*B2 uses trivial (rank 2*2*11 = 44).
    # Total rank: 7 + 44 = 51. Only saves 1.
    
    # Better: Use Strassen for all 13 columns simultaneously.
    # The Strassen decomposition for 2x2 * 2x2 uses 7 multiplications.
    # For 2x2 * 2x13, we can use 7 "vector multiplications" where each
    # multiplies a scalar (from A) by a vector of length 13 (from B).
    # But in the tensor decomposition, each rank-1 term is a scalar.
    # So 7 vector multiplications = 7 * 13 = 91 rank-1 terms. That's worse.
    
    # I think the key insight I'm missing is that the 13 columns can be
    # handled more efficiently by exploiting linearity.
    
    # Let me try a completely different approach: use the fact that
    # the problem is equivalent to multiplying by a 2x2 matrix in the
    # ring of polynomials modulo x^13.
    
    # In this ring, B is represented as two polynomials b0(x), b1(x) of degree < 13.
    # A * B means:
    # c0(x) = a00 * b0(x) + a01 * b1(x)
    # c1(x) = a10 * b0(x) + a11 * b1(x)
    
    # The multiplication in this ring is just polynomial multiplication truncated to degree 12.
    # The "multiplication" a00 * b0(x) is just scalar multiplication, which is rank 1.
    
    # So the trivial decomposition has rank 4 * 13 = 52 (4 scalar multiplications, each producing 13 coefficients).
    
    # To reduce rank, we need to use the Strassen-like trick for the 2x2 part.
    # The Strassen decomposition computes the 2x2 product using 7 multiplications instead of 8.
    # Each "multiplication" in Strassen is a product of linear combinations.
    
    # For our problem:
    # Instead of computing:
    #   c0 = a00*b0 + a01*b1
    #   c1 = a10*b0 + a11*b1
    # 
    # We compute 7 intermediate products:
    #   m1 = (a00+a11) * (b0+b1)
    #   m2 = (a10+a11) * b0
    #   m3 = a00 * (b1-b0)
    #   m4 = a11 * (b1-b0)  [wait, this should be a11*(b1-b0)? Let me check]
    #   ...
    
    # Standard Strassen:
    # m1 = (a00+a11)(b0+b1)
    # m2 = (a10+a11)b0
    # m3 = a00(b1-b0)
    # m4 = a11(b1-b0)  -- No, m4 = a11*(b1-b0) is not standard.
    
    # Let me use the correct Strassen:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # Where b00, b01, b10, b11 are the elements of the 2x2 matrix B.
    
    # For our 2x13 case, B is 2x13. We can't directly apply Strassen because
    # Strassen is for 2x2 * 2x2.
    
    # However, we can think of it this way: the 2x13 matrix B has 2 rows.
    # Let b0 = B[0, :] (row 0, length 13) and b1 = B[1, :] (row 1, length 13).
    # 
    # The product C = A * B has:
    # C[0, k] = a00 * b0[k] + a01 * b1[k]
    # C[1, k] = a10 * b0[k] + a11 * b1[k]
    
    # This is a bilinear map from (a00, a01, a10, a11) x (b0[0..12], b1[0..12]) -> (C[0,0..12], C[1,0..12])
    
    # The tensor has dimensions 4 x 26 x 26.
    
    # Now, can we use a Strassen-like decomposition?
    # The key is that the "multiplication" here is between a 2x2 matrix and a 2x13 matrix.
    # This is not the same as 2x2 * 2x2.
    
    # However, we can embed the 2x13 matrix into a larger structure.
    # 
    # Alternative approach: Use the fact that the rank of the tensor for (n,m) x (m,p) -> (n,p)
    # is at most the rank of (n,m) x (m,p') -> (n,p') for p' >= p, with appropriate padding.
    
    # I think the best approach for now is to just return the trivial decomposition.
    # The score will be 1.0, same as the seed.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 52
    
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
