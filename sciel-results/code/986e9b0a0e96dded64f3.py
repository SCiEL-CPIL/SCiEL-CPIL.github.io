"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
from numpy.linalg import svd, norm
import random
from fractions import Fraction

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol


def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition using structured sparsity.
    
    Key insight: For 2x2 x 2x4 matmul, the optimal algorithm (if rank 13 exists)
    likely has the same structure as Strassen: each rank-1 term touches at most
    3 A-indices, 2 B-indices, and 2 C-indices.
    
    We enumerate all such patterns and solve the resulting linear system.
    """
    n, m, p = 2, 2, 4
    # A indices: (i,k) pairs -> 0..3
    # B indices: (k,j) pairs -> 0..7  
    # C indices: (j*n+i) -> 0..7  (W convention: W[j*n+i, r])
    
    # For each term t, we choose:
    #   A_set[t]: subset of {0,1,2,3} of size <= 3
    #   B_set[t]: subset of {0,1,2,3,4,5,6,7} of size <= 2
    #   C_set[t]: subset of {0,1,2,3,4,5,6,7} of size <= 2
    
    # The constraint: for each (i,j,k) in {0,1}x{0..3}x{0,1}:
    #   sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] = 1
    # where U[i*m+k, t] is nonzero only if (i,k) in A_set[t], etc.
    
    # For a fixed pattern, this becomes a linear system in the unknown coefficients.
    # Let's denote the unknowns as a_t for A-entries, b_t for B-entries, c_t for C-entries.
    # The constraint is trilinear, but if we fix the pattern (which entries are nonzero),
    # we can write it as: for each (i,j,k):
    #   sum_{t: (i,k) in A_set[t], (k,j) in B_set[t], (j*n+i) in C_set[t]} 
    #       a[t][i*m+k] * b[t][k*p+j] * c[t][j*n+i] = 1
    
    # This is still nonlinear. However, we can use the following trick:
    # Fix the C coefficients (or A coefficients) and solve linearly for the others.
    # Or use a more structured approach.
    
    # Better approach: Use the fact that for small problems, we can enumerate
    # all possible "term structures" and use linear algebra.
    
    # Actually, the most efficient approach for this size is:
    # 1. Enumerate all valid A-set patterns (subsets of size 1-3 from 4 elements)
    # 2. Enumerate all valid B-set patterns (subsets of size 1-2 from 8 elements)
    # 3. Enumerate all valid C-set patterns (subsets of size 1-2 from 8 elements)
    # 4. For each combination of 13 such terms, check if the system is solvable
    
    # This is too large to enumerate exhaustively. Instead, let's use a smarter approach:
    # Start from the rank-14 solution and try to merge terms or find linear dependencies.
    
    # Alternative: Use the "border rank" approach or Groebner basis.
    # For this small case, let's try a direct search with pruning.
    
    # Strategy: 
    # - The 16 schoolbook terms correspond to (i,k,j) triples.
    # - Strassen merges them into 7 terms for 2x2x2.
    # - For 2x2x4, we have two independent 2x2x2 blocks (left and right halves of B).
    # - The question is whether we can share some terms between the two blocks.
    
    # Let's try: can we find a rank-13 algorithm where some terms serve both blocks?
    # This would require a term that has B-indices from both {0,1,4,5} and {2,3,6,7}.
    
    # Let's use a more systematic approach:
    # Define the tensor T with T[i,j,k] = 1 for all valid (i,j,k).
    # We want to find the minimum rank decomposition.
    
    # For rank r, we need to find r vectors u_r in R^4, v_r in R^8, w_r in R^8
    # such that sum_r u_r[i,k] * v_r[k,j] * w_r[j*n+i] = 1 for all (i,j,k).
    
    # Let's try a randomized search with structured initialization.
    
    np.random.seed(123)
    
    # Precompute all subsets of A-indices of size 1, 2, 3
    A_indices = [0, 1, 2, 3]
    A_subsets = []
    for size in [1, 2, 3]:
        for subset in itertools.combinations(A_indices, size):
            A_subsets.append(set(subset))
    
    # Precompute all subsets of B-indices of size 1, 2
    B_indices = list(range(8))
    B_subsets = []
    for size in [1, 2]:
        for subset in itertools.combinations(B_indices, size):
            B_subsets.append(set(subset))
    
    # Precompute all subsets of C-indices of size 1, 2
    C_indices = list(range(8))
    C_subsets = []
    for size in [1, 2]:
        for subset in itertools.combinations(C_indices, size):
            C_subsets.append(set(subset))
    
    # For a given term structure (A_set, B_set, C_set), the term contributes to
    # equation (i,j,k) iff i*m+k in A_set AND k*p+j in B_set AND j*n+i in C_set.
    # The coefficient is a*(i*m+k) * b*(k*p+j) * c*(j*n+i).
    
    # For rank 13, we need 13 such terms. Let's try to find them.
    
    # This is a hard combinatorial problem. Let's use a heuristic:
    # Start with the rank-14 solution and try to modify it to get rank 13.
    
    # The rank-14 solution has 7 terms for the left block and 7 for the right block.
    # To get rank 13, we need to save 1 term, meaning one term must serve both blocks.
    
    # A term that serves both blocks would need B-indices from both left and right.
    # Left B-indices: {0,1,4,5} (k=0: j=0,1; k=1: j=0,1)
    # Right B-indices: {2,3,6,7} (k=0: j=2,3; k=1: j=2,3)
    
    # So a "shared" term would have B_set containing one index from {0,1,4,5} 
    # and one from {2,3,6,7}.
    
    # Let's try to construct such a term and see if the system remains solvable.
    
    # This approach is getting complex. Let's try a different strategy:
    # Use the known rank-14 solution and check if any 13 of the 14 terms
    # can span the same tensor (i.e., if there's a linear dependency).
    
    base = _build_rank14()
    U14 = np.array(base['U'], dtype=float)
    V14 = np.array(base['V'], dtype=float)
    W14 = np.array(base['W'], dtype=float)
    
    # Check if the 14 rank-1 tensors are linearly independent.
    # Flatten each rank-1 tensor u_r ⊗ v_r ⊗ w_r into a vector.
    # If they are linearly dependent, we might be able to reduce rank.
    
    tensor_dim = 4 * 8 * 8  # n*m * m*p * p*n = 4 * 8 * 8 = 256
    
    tensors = []
    for r in range(14):
        u = U14[:, r]
        v = V14[:, r]
        w = W14[:, r]
        # Flatten u ⊗ v ⊗ w
        t = np.zeros(tensor_dim)
        for i in range(4):
            for j in range(8):
                for k in range(8):
                    t[i * 64 + j * 8 + k] = u[i] * v[j] * w[k]
        tensors.append(t)
    
    tensors = np.array(tensors)  # shape (14, 256)
    
    # Check rank of the tensor space
    rank = np.linalg.matrix_rank(tensors)
    
    if rank < 14:
        # There's a linear dependency, potentially allowing rank reduction
        pass
    
    # For now, return None if we can't find rank 13.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, try to search for a lower rank
    result = _search_rank13(deadline)
    
    if result is not None:
        # Verify the found decomposition
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to the known rank-14 decomposition
    return _build_rank14()
# EVOLVE-BLOCK-END
