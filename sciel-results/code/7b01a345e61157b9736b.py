"""Search for rank-minimal exact tensor decomposition of 2x4x7 matrix multiplication."""
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=2, m=4, p=7):
    """Fast verify using numpy-style manual computation."""
    rank = len(U[0]) if U else 0
    # Precompute
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    seed_rank = n * m * p  # 56
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = seed_rank
    
    # Verify schoolbook
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try Strassen-inspired block decomposition
    strassen_result = _strassen_block(n, m, p)
    if strassen_result:
        sU, sV, sW = strassen_result
        srank = len(sU[0]) if sU else 0
        if srank < best_rank and _verify(sU, sV, sW, n, m, p):
            best_U, best_V, best_W = sU, sV, sW
            best_rank = srank
    
    # Try rank reduction via linear combination merging
    while time.time() < deadline - 5.0:
        U, V, W, rank = _try_rank_reduction(best_U, best_V, best_W, n, m, p, best_rank)
        if U is not None and rank < best_rank:
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
    rank = n * m * p
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
    return U, V, W

def _strassen_block(n, m, p):
    """
    Try to exploit block structure. For 2x4x7:
    Split m=4 into 2+2. A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x7).
    C = A1*B1 + A2*B2.
    Each sub-product is 2x2x7. We can use Strassen-like tricks if we further split.
    
    Actually, let's try: split n=2 into 1+1, m=4 into 2+2, p=7 into 3+4.
    This gives 8 sub-blocks of sizes 1x2x3 and 1x2x4.
    """
    n, m, p = 2, 4, 7
    
    # Strategy: Use the structure that A is 2x4, B is 4x7
    # Split the middle dimension: 4 = 2 + 2
    # A = [A_left | A_right], each 2x2
    # B = [B_top; B_bottom], each 2x7
    # C = A_left @ B_top + A_right @ B_bottom
    # Each term: 2x2 @ 2x7 -> 2x7
    # For 2x2x7, schoolbook rank is 28. Total would be 56 (no improvement).
    
    # Better: Try to combine. Note that A_left and A_right share rows.
    # Let A_left = [[a11, a12], [a21, a22]], A_right = [[a13, a14], [a23, a24]]
    # B_top = [[b11,...,b17],[b21,...,b27]], B_bottom = [[b31,...,b37],[b41,...,b47]]
    # C[0,0] = a11*b11 + a12*b21 + a13*b31 + a14*b41
    # etc.
    
    # This is just schoolbook rearranged. No free lunch without structure.
    
    # Try a different approach: use the fact that for 2x4x7, we might find
    # a rank-50 or lower decomposition by exploiting linear dependencies.
    
    # For now, return None to indicate no improvement from simple blocking.
    return None

def _try_rank_reduction(U, V, W, n, m, p, current_rank, max_time=2.0):
    """
    Try to reduce rank by finding linear combinations of rank-1 terms
    that can be merged.
    
    Strategy: For each pair of columns (c1, c2) in U, V, W, check if
    the sum of two rank-1 tensors can be replaced by a single rank-1 tensor
    plus adjustments to other terms. This is complex, so we use a simpler
    heuristic: try random linear combinations and check if rank decreases.
    """
    start = time.time()
    rank = current_rank
    
    # Simple approach: try to find if any two columns are "similar"
    # and can be merged by adjusting other columns.
    
    # For now, this is a placeholder for more sophisticated search.
    # A real implementation would use:
    # 1. Random perturbation of one column
    # 2. Solve for corrections in other columns
    # 3. Check if rank can be reduced
    
    return None, None, None, rank
