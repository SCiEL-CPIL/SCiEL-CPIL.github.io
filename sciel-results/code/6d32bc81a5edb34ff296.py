"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # 150s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _strassen_2x2(A, B):
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    A, B are 2x2 matrices (lists of lists or numpy arrays).
    Returns C = A * B as a 2x2 matrix, and the 7 Strassen products.
    """
    # Convert to Fraction for exact arithmetic
    a = [[Fraction(A[0][0]), Fraction(A[0][1])],
         [Fraction(A[1][0]), Fraction(A[1][1])]]
    b = [[Fraction(B[0][0]), Fraction(B[0][1])],
         [Fraction(B[1][0]), Fraction(B[1][1])]]
    
    # Strassen's 7 products
    M1 = (a[0][0] + a[1][1]) * (b[0][0] + b[1][1])
    M2 = (a[1][0] + a[1][1]) * b[0][0]
    M3 = a[0][0] * (b[0][1] - b[1][1])
    M4 = a[1][1] * (b[1][0] - b[0][0])
    M5 = (a[0][0] + a[0][1]) * b[1][1]
    M6 = (a[1][0] - a[0][0]) * (b[0][0] + b[0][1])
    M7 = (a[0][1] - a[1][1]) * (b[1][0] + b[1][1])
    
    # Reconstruct C
    c = [[Fraction(0), Fraction(0)],
         [Fraction(0), Fraction(0)]]
    c[0][0] = M1 + M4 - M5 + M7
    c[0][1] = M3 + M5
    c[1][0] = M2 + M4
    c[1][1] = M1 - M2 + M3 + M6
    
    # Return the 7 products and the result
    return c, [M1, M2, M3, M4, M5, M6, M7]

def _build_strassen_decomposition(n, m, p):
    """
    Build a low-rank decomposition using Strassen's algorithm on 2x2 blocks.
    
    For 4x5 * 5x5:
    - Split A into 4 rows, each of length 5
    - Split B into 5 rows, each of length 5  
    - For each row i of A, compute C[i,:] = A[i,:] * B
    
    We decompose the 5x5 multiplication into:
    - A 2x2 block multiplication (using Strassen, rank 7)
    - Plus additional terms for the remaining dimensions
    
    Actually, let's think more carefully. The tensor T[i,j,k] for i in [0,4), j in [0,5), k in [0,5)
    needs to be decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    One approach: use the fact that we can write the multiplication as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For a fixed i, this is a 1x5 times 5x5 multiplication.
    We can decompose this using a structured approach.
    
    Let me try a different approach: use the known low-rank decompositions for small matrix multiplication.
    """
    # For now, let's try to construct a decomposition by exploiting the structure.
    # We'll use a hybrid approach: schoolbook for some parts, Strassen for others.
    
    # Actually, let me try a cleaner approach.
    # The matrix multiplication tensor for n x m * m x p has rank at most n*m*p.
    # We can try to find a lower rank decomposition by using the fact that
    # for each pair (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One idea: use the "outer product" structure.
    # For each j, we have A[:,j] (n-vector) and B[j,:] (p-vector).
    # C += A[:,j] * B[j,:]
    
    # This is n*p rank-1 updates for each j, total m*n*p = 100.
    
    # To reduce rank, we need to find linear dependencies.
    # For small dimensions, we can try to use known algorithms.
    
    # Let's try: decompose into 2x2 blocks where possible.
    # n=4 can be split into 2x2 blocks.
    # m=5 and p=5 can be split into 2x2 + 1x1 or similar.
    
    # For simplicity, let's first try to get a working decomposition with rank < 100.
    # Then optimize.
    
    # Approach: Use the fact that 4 = 2+2, and try to use Strassen on 2x2 subproblems.
    
    # Let's define a decomposition that uses:
    # - For the 4x5 part of A: split into two 2x5 blocks
    # - For B: use as is
    # - Combine using a structured bilinear algorithm
    
    # Actually, let me just implement a direct search for a low-rank decomposition
    # using the structure of the problem.
    
    # For now, return schoolbook as baseline, then we'll improve.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}, rank

def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen's algorithm on 2x2 blocks.
    
    For 4x5 * 5x5:
    - Split the 4 rows into two groups of 2
    - For each group, use a 2x5 * 5x5 multiplication
    - Decompose the 2x5 * 5x5 using a combination of Strassen and direct methods
    
    The key idea: a 2x2 * 2x2 multiplication can be done with 7 multiplications
    instead of 4. We can extend this to 2x5 * 5x5 by splitting the 5x5 into
    2x2 blocks plus remainder.
    """
    # Let's build the decomposition term by term.
    # Each term is (u, v, w) where:
    # u is a vector of length n*m = 20
    # v is a vector of length m*p = 25  
    # w is a vector of length p*n = 20
    
    # For the schoolbook, each term corresponds to a single (i,j,k) triple.
    # For Strassen, we combine multiple (i,j,k) triples into one term.
    
    # Let's try: for the 4x5 * 5x5 case, split into:
    # - Two 2x5 * 5x5 subproblems
    # For each 2x5 * 5x5:
    #   Split the 5x5 into a 2x2 block (top-left), and the rest
    #   Use Strassen on the 2x2 part
    #   Use direct multiplication for the rest
    
    # Actually, this is getting complex. Let me try a different approach.
    # Use the known result that the rank of 2x2x2 matrix multiplication is 7.
    # And build up from there.
    
    # For 4x5 * 5x5, one approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Group the j index: j = 0,1,2,3,4
    # 
    # For j=0,1: use Strassen-like combination
    # For j=2,3: use Strassen-like combination  
    # For j=4: direct
    #
    # But Strassen works on 2x2 blocks, not 1x2.
    
    # Let me try yet another approach: use the fact that we can write
    # the multiplication as a sum of outer products, and find a sparse
    # representation.
    
    # For now, let's return a decomposition with rank 100 (schoolbook)
    # and then try to optimize it.
    
    return None

def _verify_decomposition(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    # Convert to numpy for efficient computation
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    
    # All entries should be 1
    if np.max(np.abs(T - 1.0)) > 1e-9:
        return False
    
    return True

def _try_optimize(U, V, W, n, m, p, max_steps=200, lr=0.01, seed=42):
    """
    Try to reduce the rank by finding linear dependencies and eliminating terms.
    """
    rank = len(U[0])
    np.random.seed(seed)
    
    # Convert to numpy
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # Try to find a lower rank decomposition by gradient descent
    # on a subset of terms.
    
    # For simplicity, just return the original for now.
    return U, V, W, rank

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook
    schoolbook, schoolbook_rank = _build_strassen_decomposition(n, m, p)
    best = schoolbook
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # Strategy: use Strassen's algorithm on 2x2 blocks
    
    # For 4x5 * 5x5, we can try to use the following decomposition:
    # Split A into 2x2 blocks (since n=4 = 2*2)
    # For each 2x2 block of A, multiply by B using a structured algorithm
    
    # Let's try to construct a decomposition with rank < 100.
    
    # One approach: use the fact that 5 = 2 + 2 + 1
    # Split the m=5 and p=5 dimensions into 2+2+1
    # Use Strassen on the 2x2 parts
    
    # For now, let's try a direct construction.
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to find a decomposition with rank around 80-90.
    
    # Use a greedy approach: start with schoolbook, then try to merge terms.
    
    # Actually, let's try a known low-rank decomposition for small matrix multiplication.
    # For 2x2x2, the rank is 7 (Strassen).
    # We can use this as a building block.
    
    # For 4x5 * 5x5:
    # Split into 2x2 * 5x5 blocks (2 blocks in the i direction)
    # For each 2x2 * 5x5:
    #   Split 5x5 into 2x2 + 2x1 + 1x2 + 1x1 (or similar)
    #   Use Strassen on the 2x2 * 2x2 part
    #   Use direct for the rest
    
    # This is complex. Let me just try to get a working solution with rank < 100.
    
    # For now, return schoolbook. In a future iteration, we'll add the Strassen-based construction.
    
    return best

# EVOLVE-BLOCK-END
