"""Matrix multiplication 6x6x6 rank-minimal exact decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_block(r):
    """Build rank-7 Strassen decomposition for r x r x r (r must be even).
    Returns U (r^2 x 7), V (r^2 x 7), W (r^2 x 7)."""
    h = r // 2
    rank = 7
    U = np.zeros((r * r, rank))
    V = np.zeros((r * r, rank))
    W = np.zeros((r * r, rank))
    
    # Block structure: 2x2 grid of h x h blocks
    # Block (i,j) where i,j in {0,1}
    # A[i,j], B[i,j], C[i,j]
    
    # Strassen formulas for C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each rank component, we need to fill U, V, W
    # U rows: A block row index = i*h + local_row, col = j*h + local_col
    # V rows: B block row index = i*h + local_row, col = j*h + local_col
    # W rows: C block row index = i*h + local_row, col = j*h + local_col
    
    # Let's define which blocks contribute to each M
    # M1: A blocks (0,0),(1,1); B blocks (0,0),(1,1)
    # M2: A blocks (1,0),(1,1); B blocks (0,0)
    # M3: A blocks (0,0); B blocks (0,1),(1,1)
    # M4: A blocks (1,1); B blocks (1,0),(0,0)
    # M5: A blocks (0,0),(0,1); B blocks (1,1)
    # M6: A blocks (1,0),(0,0); B blocks (0,0),(0,1)
    # M7: A blocks (0,1),(1,1); B blocks (1,0),(1,1)
    
    M_blocks = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1),(0,1,1),(1,0,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),  # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,1,1)]),  # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),  # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),  # M7
    ]
    
    for m_idx, (a_blocks, b_blocks, c_blocks) in enumerate(M_blocks):
        # Fill U for this rank component
        for (bi, bj, sign) in a_blocks:
            row_start = bi * h
            col_start = bj * h
            U[row_start:row_start+h, col_start:col_start+h, :] = 0  # just to be safe
            U[row_start:row_start+h, m_idx] = sign
        
        for (bi, bj, sign) in b_blocks:
            row_start = bi * h
            col_start = bj * h
            V[row_start:row_start+h, m_idx] = sign
        
        for (bi, bj, sign) in c_blocks:
            row_start = bi * h
            col_start = bj * h
            W[row_start:row_start+h, m_idx] = sign
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    # Tensor T[i,j,k] = 1 iff U[i,c]*V[j,c]*W[k,c] sums to 1
    # Check: for each (i,j,k), sum_c U[i,c]*V[j,c]*W[k,c] == 1
    rank = U.shape[1]
    # Build tensors
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    errors += 1
                    max_err = max(max_err, abs(val - 1.0))
    return errors == 0, max_err

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    rank = U.shape[1]
    errors = 0
    max_err = 0.0
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    errors += 1
                    max_err = max(max_err, abs(val - 1.0))
    return errors == 0, max_err

def _strassen_6x6(n=6, m=6, p=6):
    """6x6x6 via 2x2 Strassen of 3x3 blocks, each 3x3 using schoolbook.
    Outer rank: 7 * (3*3*3) = 7*27 = 189.
    But we can do better: use rank-23 for 3x3? No, 3x3 min rank is 23.
    Actually, let's just use Strassen for the 2x2 partition of 6x6 into 3x3 blocks,
    where each 3x3 multiplication uses its own decomposition.
    
    Better approach: 6x6 = 2x2 blocks of 3x3.
    Use Strassen (rank 7) at the block level, each block multiplication is 3x3x3.
    If we use schoolbook for 3x3, inner rank = 27, total = 7*27 = 189.
    """
    h = 3  # block size
    # We'll build the full U, V, W matrices of sizes 36x189
    outer_rank = 7
    inner_rank = h * h * h  # 27
    total_rank = outer_rank * inner_rank
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Strassen block structure for 2x2 partition
    # A = [[A00, A01], [A10, A11]], each 3x3
    # B = [[B00, B01], [B10, B11]], each 3x3
    # C = [[C00, C01], [C10, C11]], each 3x3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M component, we need to define how A, B, C blocks map to the 3x3 multiplication
    # Each M is a 3x3 matrix product, so we need a rank-27 decomposition for 3x3x3
    # Schoolbook: 27 terms
    
    # Let's define the structure:
    # For a given M (say M1), A-part = A00 + A11, B-part = B00 + B11, C-part contributes to C00, C11
    # We'll use the same schoolbook decomposition for each 3x3 product
    
    # Index mapping for 6x6 matrices:
    # A[i,j] where i,j in 0..5
    # Block (bi, bj) covers rows bi*3..bi*3+2, cols bj*3..bj*3+2
    
    # U[i,j] -> i is row of A, j is col of A. U index = i*m + j = i*6 + j
    # V[j,k] -> j is row of B, k is col of B. V index = j*p + k = j*6 + k
    # W[k,i] -> k is row of C (which is col of B), i is col of C (which is row of A). W index = k*n + i = k*6 + i
    
    # For schoolbook 3x3: for each (a,b,c) in 0..2, term has:
    # U[a*3+b, col] = 1, V[b*3+c, col] = 1, W[c*3+a, col] = 1
    # But in the 6x6 context, the local (a,b,c) maps to global positions depending on the block
    
    # Let's build it more carefully.
    
    # Strassen components: (A_blocks, B_blocks, C_blocks)
    # Each is a list of (block_row, block_col, sign)
    S = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),   # M1 -> C00, C11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),            # M2 -> C10, C11
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),            # M3 -> C01, C11
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),            # M4 -> C00, C10
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,1,1)]),            # M5 -> C00, C01
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),            # M6 -> C11
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),            # M7 -> C00
    ]
    
    col = 0
    for m_idx, (a_blocks, b_blocks, c_blocks) in enumerate(S):
        for (a, b, c) in product(range(3), repeat=3):
            # This is one schoolbook term for the 3x3 product
            # A part: local (a, b) in each A block
            # B part: local (b, c) in each B block
            # C part: local (c, a) in each C block
            
            for (bi, bj, sign) in a_blocks:
                gi = bi * 3 + a
                gj = bj * 3 + b
                U[gi * 6 + gj, col] = sign
            
            for (bi, bj, sign) in b_blocks:
                gi = bi * 3 + b
                gj = bj * 3 + c
                V[gi * 6 + gj, col] = sign
            
            for (bi, bj, sign) in c_blocks:
                gi = bi * 3 + c
                gj = bj * 3 + a
                W[gi * 6 + gj, col] = sign
            
            col += 1
    
    return U, V, W

def _try_improve_strassen(U, V, W, n, m, p, time_budget=30):
    """Try to improve the Strassen-based decomposition by searching for lower rank.
    This is a heuristic search; may not always succeed."""
    # For now, return the input unchanged as a baseline
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline: return by 140s (10s safety margin before 150s kill)
    deadline = start_time + 140.0
    
    # Start with Strassen-based rank 189 decomposition
    U, V, W = _strassen_6x6(n, m, p)
    rank = U.shape[1]
    
    # Verify
    valid, max_err = _verify_fast(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find improvements
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Could try more advanced searches, but for now return the Strassen result
        pass
    
    # Convert to lists as required
    U_list = [[U[i, j] for j in range(rank)] for i in range(n * m)]
    V_list = [[V[i, j] for j in range(rank)] for i in range(m * p)]
    W_list = [[W[i, j] for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

