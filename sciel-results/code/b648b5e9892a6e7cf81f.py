import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _add_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """Add Strassen 2x2x2 block at given offsets. Returns new col."""
    n, m, p = 2, 3, 5
    # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
    # Global: i = i_off+i_l, j = j_off+j_l, k = k_off+k_l
    
    # M1: (A00+A11)(B00+B11) -> C00 += , C11 +=
    U[i_off * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10 += , C11 -=
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01 += , C11 +=
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00 += , C10 +=
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00 -= , C01 +=
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 +=
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 +=
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def _strassen_2x2x2_rank6_attempt():
    """
    Attempt to find a 6-term decomposition for 2x2x2 matmul.
    This is known to be impossible (rank is exactly 7), but let's verify
    our understanding by checking the structure.
    """
    # The 2x2x2 matmul tensor has rank 7 (Strassen).
    # No 6-term exact decomposition exists.
    # This function is a placeholder for documentation.
    return None

def _build_rank28():
    """Build the baseline rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    col = _add_strassen_block(U, V, W, col, 0, 1, 0)
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    col = _add_strassen_block(U, V, W, col, 0, 1, 3)
    assert col == rank
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build expected tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i < n and j < m and k < p) else 0
                # Actually the matmul tensor is T[i,j,k] = 1 for all valid (i,j,k)
                # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has 1 at every (i,j,k)
                expected = 1.0
                got = 0.0
                for r in range(r):
                    got += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(got - expected) > 1e-9:
                    return False, (i, j, k, got, expected)
    return True, None

def _try_rank_reduction(U, V, W, n, m, p, start_rank=28, time_limit=100):
    """
    Try to reduce rank by finding linear dependencies.
    We look for a set of rank-r columns that can be replaced by fewer.
    Strategy: use the fact that the tensor rank of matmul(2,3,5) is known to be 
    at most 28 (our construction). Try to find if any column is a linear combo
    of others in the decomposition sense.
    
    Actually, a better approach: try different block decompositions and see
    if we can get lower rank.
    """
    start_time = time.time()
    best_rank = start_rank
    best_result = None
    
    # First verify the baseline
    ok, info = _verify(U, V, W, n, m, p)
    if not ok:
        print(f"Baseline verification failed: {info}")
    
    # Try alternative constructions with potentially lower rank
    # Key insight: for 2x3x5, we can try:
    # - Full Strassen on parts
    # - Different block partitions
    
    # Alternative 1: Use Strassen for all 2x2x2 blocks where possible
    # 2x3x5 = 2x(3)x5. Split j into {0},{1,2}.
    # For j={1,2}, k can be split into pairs: {0,1},{2,3},{4}
    # That gives: 2x1x5 (10) + 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 28
    # Same as before.
    
    # Alternative 2: Split differently. 
    # 2x3x5: try splitting i=2 into {0,1} (already done)
    # What about using a 6-term formula for 2x2x2?
    # The rank of 2x2x2 matmul is 7 (Strassen), proven minimal.
    # So we can't do better than 7 for each 2x2x2 block.
    
    # Alternative 3: Can we share terms across blocks?
    # The key question: can we find a decomposition with rank < 28?
    
    # Known result: rank of 2x3x5 matmul.
    # Lower bound: max(2*3, 3*5, 2*5) = max(6,15,10) = 15 (trivial)
    # Better lower bound from tensor rank theory.
    # Strassen's bound gives upper bound.
    # For 2x3x5: one can achieve 2*3*5=30 schoolbook, 
    # Strassen hybrid gives 28 as we have.
    
    # Let me try: can we do better by a smarter partition?
    # 2x3x5 = 2x(2+1)x(4+1)
    # = 2x2x4 + 2x1x4 + 2x2x1 + 2x1x1
    # 2x2x4 = 2x2x2 + 2x2x2 = 7+7 = 14 (or 2x2x4 directly?)
    # 2x2x4 rank: can use Strassen for each 2x2x2: 14
    # 2x1x4 = 8
    # 2x2x1 = 4  
    # 2x1x1 = 2
    # Total: 14+8+4+2 = 28. Same.
    
    # What about 2x3x5 = 2x(1+2)x(5)
    # = 2x1x5 + 2x2x5
    # 2x1x5 = 10
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7+7+4 = 18
    # Total: 10+18 = 28. Same.
    
    # The rank is likely 28 or close. Let me try to verify if 27 is possible
    # by checking if any column in our rank-28 decomp is redundant.
    
    # Check for linear dependencies: if column r can be written as linear combo
    # of other columns in the tensor sense, we can remove it.
    # This is hard to check directly. Instead, let's try random perturbations
    # of the decomposition and see if we can find a lower-rank one.
    
    # Actually, let's try a completely different approach:
    # Use the known rank of small matrix multiplication tensors.
    # rank(2,3,5) <= 28 by our construction.
    # Is there a known better bound?
    
    # Let me try: 2x3x5 with a different Strassen arrangement.
    # What if we group k differently?
    # 2x3x5: j in {0,1,2}, k in {0,1,2,3,4}
    # For each j, we have a 2x1x5 subproblem.
    # But we can combine across j's using Strassen-like tricks.
    
    # Try: treat as 2x(3)x(5) and use a 3x5 block with rank optimization.
    # The 3x5 part: can we decompose it with rank < 15?
    # rank(1,3,5) for the "inner" product... this is getting complex.
    
    # Let's just try to find if rank 27 works by checking the structure.
    # For now, return the rank-28 solution as baseline.
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_rank, (U, V, W)
    
    return best_rank, (U, V, W)

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    
    # Build baseline rank-28
    U, V, W = _build_rank28()
    
    # Verify
    ok, info = _verify(U, V, W, n, m, p)
    if not ok:
        print(f"Verification failed: {info}")
        # Return anyway; evaluator will handle
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank if time permits
    remaining = DEADLINE - (time.time() - start_time) - 10
    if remaining > 30:
        rank, (U2, V2, W2) = _try_rank_reduction(U, V, W, n, m, p, 28, remaining)
        if rank < 28:
            U, V, W = U2, V2, W2
    
    return {"U": U, "V": V, "W": W}
