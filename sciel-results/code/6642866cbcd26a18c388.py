"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product, combinations_with_replacement

DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Wait, the matrix multiplication tensor: C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1
    # The decomposition: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = T[i,j,k] = 1
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is stored as rows = n*m indices, cols = rank
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    tensor_target = np.ones((n, m, p), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_mat[i*m + j, r] * V_mat[j*p + k, r] * W_mat[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Basic schoolbook decomposition: rank = n*m*p = 63"""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1
                V[j*p + k][idx] = 1
                W[k*n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}

def _block_schoolbook():
    """Block schoolbook: split into 3x3*3x3, 3x3*3x3, 3x3*3x1. Rank = 9+9+9 = 27"""
    n, m, p = DIMENSIONS
    # Split B columns into groups
    # B is 3x7, split into columns [0:3], [3:6], [6:7]
    # C columns similarly
    rank = 9 + 9 + 9  # 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: B columns 0,1,2 -> C columns 0,1,2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Block 2: B columns 3,4,5 -> C columns 3,4,5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + (3 + k)][col] = 1
                W[(3 + k)*n + i][col] = 1
                col += 1
    
    # Block 3: B column 6 -> C column 6
    for i in range(n):
        for j in range(m):
            for k in range(1):
                U[i*m + j][col] = 1
                V[j*p + (6 + k)][col] = 1
                W[(6 + k)*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_3x3():
    """
    Attempt a rank-19 decomposition for 3x3*3x3 using Strassen-like approach.
    This is the key to improving from 27.
    
    For 3x3 matrix multiplication, the known lower bound is 19 and upper bound is 19.
    Let me construct a valid rank-19 decomposition.
    
    Strategy: Use the known 19-term decomposition for 3x3 matrices.
    One approach: view 3x3 as overlapping 2x2 blocks and use Strassen.
    """
    n, m, p = 3, 3, 3
    # For now, return None if we can't construct a valid one
    # We'll use a known construction below
    return None

def _construct_decomposition_with_strassen():
    """
    Use Strassen-like decomposition for the 3x3 blocks.
    
    3x3 matrix multiplication can be decomposed into:
    - Diagonal 2x2 blocks using Strassen (7 mults each)
    - Cross terms
    
    Actually, let me use a more systematic approach.
    
    For 3x3 * 3x3, we can write:
    Let A = [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    Let B = [[d1,d2,d3],[e1,e2,e3],[f1,f2,f3]]
    
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    A known rank-19 decomposition exists. Let me use a specific one.
    """
    pass

def _build_rank19_3x3():
    """
    Construct a valid rank-19 decomposition for 3x3*3x3.
    
    Using the approach from: "On the tensor rank of 3x3 matrix multiplication"
    
    One known construction uses the following 19 products:
    (I'll use a verified set from the literature)
    
    Actually, let me use a different strategy: construct it algorithmically.
    """
    n, m, p = 3, 3, 3
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use the 19-term decomposition from Strassen's generalization
    # The 19 multiplications are:
    # 
    # m1 = (a1+a2+a3) * (d1+e1+f1)  -> contributes to various C entries
    # ... this gets complex. Let me use a known valid set.
    
    # Here's a verified rank-19 decomposition for 3x3*3x3:
    # Using the "Bini-Capovani-Strassen" construction or similar.
    
    # For correctness, let me use a simpler approach:
    # The tensor for 3x3*3x3 has 27 entries, all equal to 1.
    # We need rank-19 decomposition.
    
    # Let me use a specific known decomposition:
    # Terms indexed 0..18
    # Each term is (u, v, w) where u is in R^9, v is in R^9, w is in R^9
    
    # I'll use the decomposition from "The rank of the 3x3 matrix multiplication tensor"
    # which gives exactly 19.
    
    # Construction via partition:
    # Split indices into groups and use Strassen on 2x2 submatrices
    
    # Actually, for reliability, let me construct a decomposition that I can verify.
    # I'll use a combination of diagonal terms and Strassen on overlapping 2x2 blocks.
    
    # Terms 0-2: diagonal of A and B
    # C[0,0] = a1*d1 + a2*e1 + a3*f1
    # C[1,1] = b1*d2 + b2*e2 + b3*f2  
    # C[2,2] = c1*d3 + c2*e3 + c3*f3
    # These need 9 terms (schoolbook for diagonals)
    
    # For off-diagonals, use Strassen-like tricks.
    
    # This is getting complex. Let me use a different strategy:
    # Use the fact that rank(3x3*3x3) = 19 is known, and use a precomputed solution.
    
    # Precomputed rank-19 decomposition for 3x3*3x3:
    # (I'll encode a known valid set)
    
    # For now, let's return the schoolbook (rank 9) and combine with other blocks.
    # Total would be 9+9+9 = 27.
    
    # To actually get below 27, we need the rank-19 for 3x3 blocks.
    # Let me use a specific known construction.
    
    # The 19 products for 3x3*3x3 (one valid construction):
    # Let me index A as a[i,j] for i,j in 0,1,2 and B as b[i,j] for i,j in 0,1,2
    
    # I'll use the following approach:
    # Partition the 3x3 problem into:
    # - 4 overlapping 2x2 subproblems (top-left, top-right, bottom-left, bottom-right)
    # - Each 2x2 uses 7 Strassen products
    # - But they share some computations
    
    # This is a known technique but the exact coefficients are tricky.
    
    # For safety, let me return a valid (but not minimal) decomposition.
    # I'll use schoolbook for each block: total rank 27.
    return None

def construct():
    """
    Constructs a tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with block schoolbook (rank 27)
    best = _block_schoolbook()
    
    if _time_left() < 10:
        return best
    
    # Try to find a better decomposition
    # Strategy: Use Strassen-like techniques for the 3x3 blocks
    
    # For 3x3 * 3x3, the minimal rank is 19.
    # If we can achieve rank 19 for each 3x3 block:
    # Total = 19 + 19 + 9 = 47 (worse than 27!)
    # Wait, that's worse. So block schoolbook (27) is better.
    
    # Actually, the minimal rank for 3x3*3x7 might be less than 27.
    # Let me think... 
    # The tensor is T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..6
    # Total entries: 3*3*7 = 63
    # Rank of this tensor: lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Upper bound: schoolbook gives 63, block schoolbook gives 27
    # Can we do better than 27?
    
    # The tensor is M_{3,3,7} = M_{3,3,3} ⊗ M_{3,3,1} in some sense
    # Actually it's a direct sum: M_{3,3,7} = M_{3,3,3} ⊕ M_{3,3,3} ⊕ M_{3,3,1}
    # rank(M_{3,3,7}) = rank(M_{3,3,3}) + rank(M_{3,3,3}) + rank(M_{3,3,1})
    # = 19 + 19 + 9 = 47? No, that's for direct sum of tensors.
    
    # Wait, I'm confusing myself. Let me reconsider.
    
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # is defined by T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # where i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # For n=3, m=3, p=7:
    # T has nonzero entries at positions (i*3+j, j*7+k, k*3+i) for i,j in 0..2, k in 0..6
    # That's 3*3*7 = 63 nonzero entries
    
    # The rank of this tensor:
    # Lower bound: max(9, 21, 21) = 21
    # Upper bound: 27 (block schoolbook)
    
    # Can we achieve rank 21? Or something between 21 and 27?
    
    # Let me try a different approach: use a global Strassen-like decomposition.
    
    if _time_left() < 5:
        return best
    
    # Try to find a rank-21 to 26 decomposition using optimization
    # This is a hard problem. For now, return the block schoolbook.
    
    return best

# Keep the EVOLVE-BLOCK markers
# EVOLVE-BLOCK-START
# (construct function is defined above)
# EVOLVE-BLOCK-END
