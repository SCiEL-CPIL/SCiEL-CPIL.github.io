"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations
from collections import defaultdict

_start_time = time.time()
TIME_BUDGET = 120.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(n1, n2, m1, m2, p1, p2, offset_i=0, offset_j=0, offset_k=0):
    """
    Apply Strassen's 2x2x2 algorithm to a sub-block of the multiplication.
    Returns (U_contrib, V_contrib, W_contrib) as sparse dicts mapping
    (index, rank_col) -> value.
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # Products:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    U = {}  # (i, j) -> {rank_col: val}
    V = {}  # (j, k) -> {rank_col: val}
    W = {}  # (k, i) -> {rank_col: val}
    
    base_col = 0
    
    def add_term_U(i, j, col, val):
        if val != 0:
            U.setdefault((i, j), {})[col] = val
    
    def add_term_V(j, k, col, val):
        if val != 0:
            V.setdefault((j, k), {})[col] = val
    
    def add_term_W(k, i, col, val):
        if val != 0:
            W.setdefault((k, i), {})[col] = val
    
    # The 7 Strassen products
    # Each product involves sums/differences of elements
    # We need to represent these as rank-1 contributions
    
    # P1 = (A11+A22)*(B11+B22)
    # This contributes to C11, C12, C21, C22
    col = base_col
    add_term_U(offset_i+0, offset_j+0, col, 1)
    add_term_U(offset_i+1, offset_j+1, col, 1)
    add_term_V(offset_j+0, offset_k+0, col, 1)
    add_term_V(offset_j+0, offset_k+1, col, 1)
    add_term_V(offset_j+1, offset_k+0, col, 1)
    add_term_V(offset_j+1, offset_k+1, col, 1)
    add_term_W(offset_k+0, offset_i+0, col, 1)
    add_term_W(offset_k+1, offset_i+0, col, 1)
    add_term_W(offset_k+0, offset_i+1, col, 1)
    add_term_W(offset_k+1, offset_i+1, col, 1)
    base_col += 1
    
    # P2 = (A21+A22)*B11
    col = base_col
    add_term_U(offset_i+1, offset_j+0, col, 1)
    add_term_U(offset_i+1, offset_j+1, col, 1)
    add_term_V(offset_j+0, offset_k+0, col, 1)
    add_term_W(offset_k+0, offset_i+1, col, 1)
    base_col += 1
    
    # P3 = A11*(B12-B22)
    col = base_col
    add_term_U(offset_i+0, offset_j+0, col, 1)
    add_term_V(offset_j+0, offset_k+1, col, 1)
    add_term_V(offset_j+1, offset_k+1, col, -1)
    add_term_W(offset_k+1, offset_i+0, col, 1)
    base_col += 1
    
    # P4 = A22*(B21-B11)
    col = base_col
    add_term_U(offset_i+1, offset_j+1, col, 1)
    add_term_V(offset_j+1, offset_k+0, col, 1)
    add_term_V(offset_j+0, offset_k+0, col, -1)
    add_term_W(offset_k+0, offset_i+1, col, 1)
    base_col += 1
    
    # P5 = (A11+A12)*B22
    col = base_col
    add_term_U(offset_i+0, offset_j+0, col, 1)
    add_term_U(offset_i+0, offset_j+1, col, 1)
    add_term_V(offset_j+1, offset_k+1, col, 1)
    add_term_W(offset_k+1, offset_i+0, col, 1)
    base_col += 1
    
    # P6 = (A21-A11)*(B11+B12)
    col = base_col
    add_term_U(offset_i+1, offset_j+0, col, 1)
    add_term_U(offset_i+0, offset_j+0, col, -1)
    add_term_V(offset_j+0, offset_k+0, col, 1)
    add_term_V(offset_j+0, offset_k+1, col, 1)
    add_term_W(offset_k+0, offset_i+1, col, 1)
    add_term_W(offset_k+1, offset_i+1, col, 1)
    base_col += 1
    
    # P7 = (A12-A22)*(B21+B22)
    col = base_col
    add_term_U(offset_i+0, offset_j+1, col, 1)
    add_term_U(offset_i+1, offset_j+1, col, -1)
    add_term_V(offset_j+1, offset_k+0, col, 1)
    add_term_V(offset_j+1, offset_k+1, col, 1)
    add_term_W(offset_k+0, offset_i+0, col, 1)
    add_term_W(offset_k+1, offset_i+0, col, 1)
    base_col += 1
    
    return U, V, W, base_col

def _build_from_sparse(U_sparse, V_sparse, W_sparse, n, m, p, rank):
    """Convert sparse dicts to dense matrices."""
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for (i, j), cols in U_sparse.items():
        for col, val in cols.items():
            U[i*m+j][col] += val
    
    for (j, k), cols in V_sparse.items():
        for col, val in cols.items():
            V[j*p+k][col] += val
    
    for (k, i), cols in W_sparse.items():
        for col, val in cols.items():
            W[k*n+i][col] += val
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_blocks():
    """
    Try to build a low-rank decomposition using Strassen blocks.
    For 3x3 x 3x8:
    - Split B into four 3x2 blocks
    - Use Strassen on overlapping 2x2 sub-blocks of A
    """
    n, m, p = 3, 3, 8
    
    # Strategy: Split the 8 columns into four groups of 2
    # For each group of 2 columns, we can potentially use shared Strassen structure
    # A is 3x3, so we can use a 2x2 Strassen on the top-left 2x2, and handle
    # the remaining row/col separately
    
    # Actually, let's try a different approach:
    # Use 4 Strassen blocks, each handling 2 columns of B
    # Each Strassen block uses 7 multiplications
    # But we need to handle the 3x3 A properly
    
    # Better approach: Pad A to 4x4, use Strassen recursively
    # Or: use the fact that 3 = 2+1, so split into 2x2 + 1x1
    
    # Let's try: split A into [[A11, A12], [A21, A22]] where A11 is 2x2,
    # A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Similarly for B and C
    
    # This gets complex. Let's try a simpler approach first:
    # Just use schoolbook but with some sharing
    
    # For now, let's try the 4-Strassen-block approach from the knowledge:
    # Embed 4 non-overlapping Strassen blocks
    
    U_sparse = defaultdict(lambda: defaultdict(float))
    V_sparse = defaultdict(lambda: defaultdict(float))
    W_sparse = defaultdict(lambda: defaultdict(float))
    
    col_offset = 0
    
    # Split B's 8 columns into 4 groups of 2
    # For each group, apply a modified Strassen to the 3x3 A and 3x2 B_block
    
    # Actually, let's try a cleaner approach:
    # Use 2x2 Strassen on the top-left 2x2 submatrix of A
    # Handle the rest with schoolbook terms
    
    # Top-left 2x2 of A: rows 0-1, cols 0-1
    # This handles C[0:2, :] = A[0:2,:] @ B[:, :]
    # But we only handle B[:, 0:2] with Strassen, rest with schoolbook
    
    # Let me try a different strategy: direct combinatorial search
    # for a rank-64 decomposition using matching
    
    return None

def _combinatorial_search(n, m, p, target_rank):
    """
    Try to find a decomposition of the given rank using combinatorial methods.
    This is a simplified approach - we'll try to find a valid decomposition
    by constructing it directly.
    """
    # For 3x3 x 3x8, schoolbook is 72
    # We want to find something below 72
    
    # One approach: use the structure of the problem
    # C = A @ B where A is 3x3 and B is 3x8
    
    # We can write this as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has entries T[i,j,k] = 1 if we're computing C[i,k]
    # A rank-1 decomposition is: sum_c u_c[i] * v_c[j] * w_c[k]
    
    # For a valid decomposition, we need:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # This is a system of 3*3*8 = 72 equations
    
    # Let's try to construct a solution with rank r < 72
    
    # One idea: use symmetry. The 8 columns of B can be grouped.
    # If we can find a decomposition where the W factors for different k
    # are related, we might save rank.
    
    # For now, return None to indicate we haven't found a better decomposition
    return None

def _try_rank_64():
    """Try to construct a rank-64 decomposition."""
    n, m, p = 3, 3, 8
    r = 64
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Use the fact that we can group the 8 columns
    # B has shape 3x8, split into two 3x4 blocks
    # A@B1 and A@B2 each need 3*3*4=36 mults in schoolbook
    # But we can potentially share: A is the same, so compute intermediate values
    # Let M1 = A @ e_j for j in 0..2 (3 vectors of dim 3)
    # Then C[:, 4l:t] = M1 @ B1[:, l] etc.
    # This doesn't directly help with rank reduction in the tensor sense.
    
    # Alternative: use a 2x2 Strassen-like block decomposition
    # Split A into 2x2 blocks (with one row/col padded), B into 2x2 blocks
    # This gets complicated for 3x3 x 3x8.
    
    # For now, return None to indicate failure
    return None

def _try_strassen_improved():
    """
    Try to build a low-rank decomposition using Strassen blocks.
    Based on the knowledge: embedding 4 non-overlapping Strassen blocks
    reduces rank from 72 to 68.
    """
    n, m, p = 3, 3, 8
    
    U_sparse = defaultdict(lambda: defaultdict(float))
    V_sparse = defaultdict(lambda: defaultdict(float))
    W_sparse = defaultdict(lambda: defaultdict(float))
    
    col_offset = 0
    
    # Strategy: Split the 8 columns into 4 groups of 2
    # For each group, use a 2x2 Strassen on a 2x2 sub-block of A
    # and handle the remaining parts with schoolbook
    
    # Actually, let's try a different approach based on the knowledge:
    # Use 4 Strassen blocks, each handling a portion of the computation
    
    # For 3x3 A, we can think of it as a 2x2 block (top-left) plus
    # a cross term and a bottom-right element
    
    # Let me try: for each pair of columns in B, use Strassen on
    # the 2x2 top-left of A, and add schoolbook terms for the rest
    
    # This is getting complex. Let's just return the schoolbook for now
    # and rely on the _verify function to check correctness
    
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = best["U"][0].__len__()
    
    # Try various approaches
    candidates = []
    
    # Try Strassen-based improvements
    if _time_left() > 5.0:
        cand = _try_strassen_improved()
        if cand is not None:
            candidates.append(cand)
    
    # Try combinatorial search
    if _time_left() > 5.0:
        cand = _combinatorial_search(n, m, p, 64)
        if cand is not None:
            candidates.append(cand)
    
    # Try rank-64
    if _time_left() > 5.0:
        cand = _try_rank_64()
        if cand is not None:
            candidates.append(cand)
    
    for cand in candidates:
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            r = cand["U"][0].__len__()
            if r < best_rank:
                best = cand
                best_rank = r
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
