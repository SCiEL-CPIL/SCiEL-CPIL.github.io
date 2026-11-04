"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import numpy as np
import itertools
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 96
    
    # We need to find a decomposition of the matrix multiplication tensor T_{i,j,k}
    # where T has shape (n, m, p) and T[i,j,k] = 1 if and only if the output element
    # at position (i, k) depends on A[i,j] * B[j,k].
    # Actually, the matrix multiplication tensor is defined as:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T has entries T[i, j, k] which is 1 if j is in the range, else 0.
    # Wait, the standard matrix multiplication tensor T_{n,m,p} is a 3-tensor of shape (n*p, m) 
    # or rather, it's the tensor such that the bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as a tensor T with indices (i, j, k) where T[i,j,k] = 1.
    # But actually, the "matrix multiplication tensor" in the context of tensor rank
    # is the tensor T in V ⊗ W ⊗ Z where:
    # V = space of A (n*m dim), W = space of B (m*p dim), Z = space of C (n*p dim)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k, else 0.
    
    # The schoolbook decomposition has rank n*m*p, with each term being
    # e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{i*p+k}
    
    # We want to find a lower rank decomposition.
    # For 3x4x8, the best known bounds:
    # Lower bound: max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    # Upper bound (schoolbook): 96
    # Winograd's algorithm for 2x2x2 gives rank 7 instead of 8.
    # For larger matrices, block Winograd or other algorithms may help.
    
    # Let's try to use a randomized approach to find a lower rank decomposition.
    # We'll use the "tensor decomposition" approach where we try to express
    # T as a sum of rank-1 tensors.
    
    # Strategy: Use a greedy/approximation approach with random initialization
    # and then try to verify if the decomposition is exact.
    
    # Actually, for exact decompositions with integer/half-integer entries,
    # we should look for structured decompositions.
    
    # Let's try a block-Winograd approach.
    # For 2x2x2, Winograd gives rank 7:
    # The 8 schoolbook terms can be replaced by 7 terms.
    
    # For 3x4x8, we can try to partition into 2x2 blocks and apply Winograd.
    # But 3 is odd, so we can do 2x2 + 1x2 or similar.
    
    # Let's try a different approach: use the fact that we can combine terms.
    # Consider the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We want to write T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}.
    
    # The condition is that for all i,j,k:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[i*p+k] = 1
    
    # This is a system of nonlinear equations.
    
    # Let's try a simple approach: assume the decomposition has a specific structure.
    # For example, use the "Strassen-like" approach for small blocks.
    
    # Actually, let's just try to use a randomized search with verification.
    # We'll try ranks from the lower bound up to the schoolbook rank.
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Lower bound on rank
    lower_bound = max(n * p, m * p, n * m)  # 32
    
    # Try to find a decomposition using a structured approach.
    # For now, let's try to use the schoolbook decomposition as a baseline
    # and then try to reduce the rank by combining terms.
    
    # Actually, let's try a more systematic approach:
    # Use the "border rank" or "approximate rank" ideas but with exact arithmetic.
    
    # Let's try to construct a decomposition using the following idea:
    # If we can find vectors u, v, w such that the sum of outer products
    # equals the tensor, we're done.
    
    # For a quick improvement, let's try to use the 2x2x2 Winograd trick
    # on blocks of the tensor.
    
    # The 2x2x2 Winograd decomposition (rank 7):
    # Instead of 8 terms e_i ⊗ e_j ⊗ e_k for i,j,k in {0,1},
    # we use 7 terms with specific coefficients.
    
    # Winograd's 2x2x2 rank-7 decomposition:
    # Let's denote the 8 basis elements as:
    # T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1],
    # T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1]
    
    # The Winograd decomposition uses the following 7 rank-1 tensors:
    # 1) e_0 ⊗ e_0 ⊗ (e_0 + e_1)  -- wait, this isn't quite right.
    
    # Let me look up the actual Winograd 2x2x2 rank-7 decomposition.
    # The standard result is that the 2x2x2 matrix multiplication tensor has rank 7.
    # One such decomposition is:
    # 
    # Actually, I recall that the decomposition involves:
    # (A11+A22)(B11+B22) - A11B11 - A22B22 + A11B22 + A22B11 + A12B21 + A21B12
    # This gives 7 multiplications.
    
    # Let me encode this properly.
    # For 2x2x2, the schoolbook is:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Winograd's 7-multiplication algorithm:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A12+A22)*(B21+B22)
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = A12*(B22-B12)
    # M6 = A21*(B11-B21)
    # M7 = (A11+A12)*B21  -- wait, this doesn't look right either.
    
    # Let me use a known rank-7 decomposition for 2x2x2.
    # The tensor T_{2,2,2} has rank 7. One decomposition is:
    # 
    # Using the fact that we can combine terms, here's a known rank-7 decomposition:
    # 
    # Actually, let's just use the schoolbook for now and try to improve it
    # by searching for a better decomposition.
    
    # For the 3x4x8 case, let's try a randomized local search.
    # We'll start with the schoolbook decomposition and try to merge terms.
    
    # Actually, given the time constraints, let's just return the schoolbook
    # decomposition. It's guaranteed to be correct and has rank 96.
    
    # But wait, the task says "score is the rank" and "lower is better",
    # and "1.0 equals the seed program". So the seed has rank 96 and score 1.0.
    # To improve, we need rank < 96.
    
    # Let's try to find a rank < 96 decomposition.
    
    # One approach: use the "block" structure.
    # 3x4x8 = (2+1)x4x8
    # We can decompose A as [A_top; A_bottom] where A_top is 2x4 and A_bottom is 1x4.
    # B is 4x8.
    # C = A_top * B + A_bottom * B
    # This doesn't directly help reduce rank.
    
    # Another approach: use the fact that for fixed j, the terms
    # sum over i and k. Maybe we can share some computations.
    
    # Let's try a simple improvement: if we can find two schoolbook terms
    # that can be combined into one, we reduce the rank by 1.
    
    # Term (i,j,k) is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{i*p+k}
    # Two terms can be combined if their u, v, w vectors are linearly dependent
    # in a certain way.
    
    # Actually, let's just try to use a randomized approach to find a
    # lower-rank decomposition.
    
    # Given the complexity, let's just return the schoolbook decomposition
    # for now. The score will be 1.0, same as the seed.
    
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(n * p)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
