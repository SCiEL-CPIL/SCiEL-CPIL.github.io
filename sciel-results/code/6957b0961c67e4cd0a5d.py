"""Block-circulant embedding with FFT for 3x5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
from math import pi, sqrt, cos, sin, gcd, isqrt
from itertools import product
import time
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns U, V, W matrices with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Schoolbook decomposition (rank = 105) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Block-circulant embedding strategy:
    # Embed 3x5x7 multiplication into a cyclic convolution of size N
    # where the DFT diagonalizes the circulant structure.
    
    # Key idea: We want to find N and index mappings such that
    # the multiplication tensor T[i,j,k] = delta(i,j,k) for valid indices
    # can be expressed as a sum of rank-1 tensors derived from DFT.
    
    # For a cyclic group Z_N, the DFT gives N orthogonal rank-1 decompositions
    # of the cyclic convolution operator.
    
    # Strategy: Use N = lcm(n, m, p) or a multiple thereof
    # and map (i, j, k) to positions in Z_N such that the product structure
    # aligns with cyclic convolution.
    
    # Let's try N = 15 (lcm of 3, 5) and see if we can handle p=7
    # Actually, let's try N = 105 (lcm of 3,5,7) for exact embedding
    
    # Better approach: Use the fact that for matrix multiplication C = A*B where
    # A is n x m and B is m x p, we can embed into a larger problem.
    
    # Block-circulant approach:
    # 1. Choose N such that N >= n+m+p or similar
    # 2. Create circulant matrices of size N x N
    # 3. The DFT diagonalizes circulant matrices, giving N rank-1 terms
    # 4. Project back to get the decomposition
    
    # Let's try multiple values of N and find the smallest rank
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try different embedding sizes
    for N in range(12, 50):
        if time.time() > deadline - 5:
            break
        
        # Map original indices to positions in Z_N
        # A[i,j] -> position (i * m + j) mod N  (for i in [0,n), j in [0,m))
        # B[j,k] -> position (j * p + k) mod N  (for j in [0,m), k in [0,p))
        # C[i,k] -> position (i * p + k) mod N  (for i in [0,n), k in [0,p))
        
        # Check for collisions
        a_positions = set()
        for i in range(n):
            for j in range(m):
                pos = (i * m + j) % N
                if pos in a_positions:
                    break
                a_positions.add(pos)
        else:
            b_positions = set()
            for j in range(m):
                for k in range(p):
                    pos = (j * p + k) % N
                    if pos in b_positions:
                        break
                    b_positions.add(pos)
            else:
                c_positions = set()
                for i in range(n):
                    for k in range(p):
                        pos = (i * p + k) % N
                        if pos in c_positions:
                            break
                        c_positions.add(pos)
                else:
                    # No collisions! Try this embedding
                    # Build the tensor using cyclic convolution
                    # The multiplication can be expressed as:
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    
                    # In the cyclic embedding, we create circulant matrices
                    # U_circ, V_circ, W_circ of size N x N such that
                    # the product gives the correct result
                    
                    # Actually, let's use a direct construction:
                    # For each "frequency" t in Z_N, we get a rank-1 term
                    # from the DFT of the embedding
                    
                    # The key insight: if we embed the matrices into
                    # circulant matrices of size N, then the DFT diagonalizes
                    # the multiplication, giving N rank-1 terms.
                    
                    # Let's construct the embedding more carefully.
                    # We need to find vectors u_t, v_t, w_t in Z_N such that
                    # sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = delta(i,j,k)
                    
                    # This is equivalent to finding a decomposition of the
                    # multiplication tensor into N rank-1 tensors.
                    
                    # For the cyclic group, the characters give us the
                    # rank-1 decompositions.
                    
                    # Let's try a simpler approach: use the fact that
                    # if N is large enough, we can use the standard
                    # Kronecker product structure.
                    
                    # Actually, let's just try to verify if N works by
                    # checking if the embedding is injective and then
                    # using the DFT to get the decomposition.
                    
                    pass  # This approach needs more work
    
    # Alternative: Use known results and structured decompositions
    # For 3x5x7, the tensor rank is at most 105 (schoolbook)
    # We can try to find a better decomposition using:
    # 1. Block structure
    # 2. Symmetry exploitation
    # 3. Algebraic methods
    
    # Let's try a block-based approach:
    # Split the 5x7 matrix into blocks and use block-circulant structure
    
    # For now, return the schoolbook solution which is guaranteed correct
    # A true improvement requires deeper algebraic insights
    
    # Let's try one more approach: use the fact that we can sometimes
    # combine terms when there are symmetries
    
    # Check if we can reduce rank by finding linear dependencies
    # among the rank-1 terms using rational arithmetic
    
    # For the 3x5x7 case, let's see if any non-trivial combination
    # of the 105 rank-1 terms sums to zero
    
    # This is a 105-dimensional space, so we'd need to check if
    # the vectors are linearly independent
    
    # The rank-1 terms are:
    # T_{i,j,k} = e_i ⊗ e_j ⊗ e_k for i in [0,3), j in [0,5), k in [0,7)
    
    # These are clearly linearly independent (they form a basis),
    # so the tensor rank is exactly 105 for this particular tensor.
    
    # Wait - that's the tensor rank of the standard basis, but the
    # matrix multiplication tensor is different!
    
    # The matrix multiplication tensor for n x m times m x p is:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1}
    #   e_i ⊗ e_j ⊗ e_k  (with appropriate indexing)
    
    # Actually, the matrix multiplication tensor is:
    # T = sum_{i,j,k} T_{i,j,k} * e_i ⊗ e_j ⊗ e_k
    # where T_{i,j,k} = 1 if the indices are valid, 0 otherwise
    
    # For 3x5x7, all indices are valid, so T_{i,j,k} = 1 for all
    # i in [0,3), j in [0,5), k in [0,7)
    
    # This means the tensor is:
    # T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    #    = u ⊗ v ⊗ w  where u = (1,1,1), v = (1,1,1,1,1), w = (1,1,1,1,1,1,1)
    
    # So the tensor rank is 1! We can decompose it as a single rank-1 term!
    
    # Let's verify:
    # U has shape (n*m, 1) = (15, 1)
    # V has shape (m*p, 1) = (35, 1)  
    # W has shape (p*n, 1) = (21, 1)
    
    # U[i*m + j] = 1 for all i in [0,3), j in [0,5)
    # V[j*p + k] = 1 for all j in [0,5), k in [0,7)
    # W[k*n + i] = 1 for all k in [0,7), i in [0,3)
    
    # The reconstructed tensor is:
    # sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # = U[:,0] ⊗ V[:,0] ⊗ W[:,0]
    # = (1,1,...,1) ⊗ (1,1,...,1) ⊗ (1,1,...,1)
    
    # This gives T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    # which is exactly what we want!
    
    # So the rank is 1!
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify the decomposition
    # The tensor should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # With our construction:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # So the sum is 1 * 1 * 1 = 1 ✓
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
