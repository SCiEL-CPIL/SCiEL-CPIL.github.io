"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

def _construct_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can verify by checking all (i,j,k) entries
    # T[i,j,k] = 1 if the indices are valid (which they always are)
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # Let's verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank = 99 (better than schoolbook 125)
    # Use a structured approach based on matrix multiplication decomposition
    # The key insight: we can use the fact that matrix multiplication has structure
    # We'll try to find a low-rank decomposition using the algebraic structure
    
    # Start with schoolbook and try to compress
    # Rank 125 is trivial. Let's aim for lower.
    
    # For 5x5 matrix multiplication, the best known upper bound is around 99
    # (Winograd's scheme adapted to 5x5)
    
    # Winograd's approach for 5x5:
    # Split 5x5 into blocks and use recursive/structured decomposition
    # 
    # For a 5x5 matrix product C = A * B:
    # We can use the fact that 5 = 2 + 3
    # Split into 2x2, 2x3, 3x2, 3x3 blocks
    # Use Winograd's scheme for each block multiplication
    #
    # Actually, let's use a more direct approach:
    # The tensor rank of 5x5 matrix multiplication
    # 
    # Known results:
    # - Schoolbook: 125
    # - Winograd's: can do better
    # - Best known: 99 (Strassen-like improvements)
    
    # Let's construct a decomposition of rank 99 using block structure
    # 
    # Approach: Use the fact that we can compute C = AB using fewer than 125 multiplications
    # by using sums and differences (Strassen-style)
    
    # For 5x5, one approach:
    # Partition A into blocks A11(2x2), A12(2x3), A21(3x2), A22(3x3)
    # Partition B into blocks B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    # 
    # C11 = A11*B11 + A12*B21 (2x2 * 2x2 + 2x3 * 3x2)
    # C12 = A11*B12 + A12*B22 (2x2 * 2x3 + 2x3 * 3x3)
    # C21 = A21*B11 + A22*B21 (3x2 * 2x2 + 3x3 * 3x2)
    # C22 = A21*B12 + A22*B22 (3x2 * 2x3 + 3x3 * 3x3)
    
    # Each sub-product can be computed with fewer multiplications
    # 
    # Let's count:
    # 2x2*2x2: rank 7 (Strassen)
    # 2x3*3x2: This is a 2x2 result, each entry is a dot product of length 3
    #   Actually, 2x3 * 3x2 gives a 2x2 matrix. Each entry is a sum of 3 products.
    #   We can use rank 3 (schoolbook) or maybe less?
    #   For 2x3*3x2 -> 2x2, the tensor has dimension (2*3, 3*2, 2*2) = (6,6,4)
    #   Schoolbook rank = 6. Can we do better?
    #   Actually this is just 4 dot products of length 3, so rank 4 if we use the structure
    #   Wait, no. Each output entry C[i,j] = sum_k A[i,k]*B[k,j] for k=0..2
    #   So we need 4 multiplications (one for each (i,j) pair), each involving 3 terms
    #   But we can share: total rank = 4 (each output is one term in the decomposition)
    #   Actually no, each output C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
    #   This is a sum of 3 terms per output, and there are 4 outputs
    #   So schoolbook rank = 4 * 3 = 12
    #   But we can't really reduce below 4 since we need 4 independent outputs
    #   Hmm, actually the tensor rank might be lower.
    
    # This is getting complex. Let me try a different approach.
    
    # Use the known result: 5x5 matrix multiplication can be done with rank 99
    # Let's construct it using the Winograd scheme.
    
    # Actually, let's try to use a simple improvement:
    # The schoolbook decomposition has rank 125.
    # We can use the fact that some entries of U, V, W can be combined.
    
    # Alternative: use the identity that (a+b)(c+d) = ac + ad + bc + bd
    # to combine terms.
    
    # For 5x5, let's try to construct rank 99 manually.
    # This is complex, so let's start with a simpler improvement.
    
    # Rank 119: Use Strassen for the 4x4 subproblem
    # Split 5x5 into 4x4 + 1x1 corners
    # Compute 4x4 product with Strassen (rank 18 instead of 16 schoolbook for 4x4? No, Strassen is rank 7 for 2x2, so 4x4 is rank 7^2 = 49, schoolbook is 16)
    # Wait, Strassen for 2x2 is rank 7, schoolbook is 4. So Strassen is worse for 2x2!
    # Strassen is rank 7 for 2x2 vs schoolbook 4. That's worse.
    # Oh right, Strassen reduces the number of multiplications from 8 to 7 for 2x2, but schoolbook is 4 multiplications for 2x2.
    # I'm confusing myself. Let me restart.
    
    # For 2x2 matrix multiplication:
    # C = AB where A, B are 2x2
    # Schoolbook: 4 multiplications (each C[i,j] = sum_k A[i,k]*B[k,j])
    # Wait no, that's 4 sums of 2 products each, so 8 scalar multiplications total
    # But in terms of tensor rank, each C[i,j] is a bilinear form
    # The tensor for 2x2 matrix multiplication has rank 7 (Strassen)
    # Schoolbook rank is 4 (one term for each (i,j) pair, each term being a product of a row of A and column of B)
    # Wait, that's not right either.
    
    # Let me think carefully:
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # where i indexes rows of A, j indexes columns of B (which are rows of B), k indexes...
    # Actually, T[i,j,k] means: output C[i,k] gets contribution A[i,j]*B[j,k]
    # So T is a 3-tensor of size (n*m, m*p, p*n) or equivalently indexed by (i,j,k)
    # 
    # For n=m=p=2:
    # T[i,j,k] for i,j,k in {0,1}
    # T has 8 non-zero entries (all 8)
    # Schoolbook: rank 4, with terms (i,j,k) where i=j=k... no
    # Schoolbook decomposition: for each (i,k) pair, C[i,k] = sum_j A[i,j]*B[j,k]
    # This gives 4 terms, each involving all j values
    # But in the tensor decomposition, each term is a product of 3 vectors
    # Term (i,k): U[i, c] * V[k, c] * W[?, c]
    # Hmm, I need to be more careful.
    
    # The standard decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: for each (i,j,k), T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k]
    # In the tensor decomposition framework:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i*m+j, k*p+l, ...] no this is getting confused.
    
    # Let me use the convention from the task:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So U is indexed by (i,j) where i in [0,n), j in [0,m)
    # V is indexed by (j,k) where j in [0,m), k in [0,p)
    # W is indexed by (k,i) where k in [0,p), i in [0,n)
    # 
    # The product sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should give the tensor
    # For matrix multiplication, this should equal 1 for all valid (i,j,k)
    # 
    # Schoolbook: for each (i,j,k), set U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # This gives rank n*m*p = 125 for 5x5x5
    
    # To reduce rank, we need to find linear dependencies or use algebraic identities.
    
    # For now, let's just return the schoolbook decomposition but verify it works.
    # Then in future iterations, we can try to improve.
    
    rank = n * m * p  # Start with schoolbook
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1.0
                V[j*p+k][column] = 1.0
                W[k*n+i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
