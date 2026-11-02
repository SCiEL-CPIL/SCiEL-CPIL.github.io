"""Matrix multiplication 2x2x9 - tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposed tensor
    D = np.einsum('ar,br,cr->abc', U, V, W)
    
    max_err = np.max(np.abs(D - T))
    return max_err < 1e-10, max_err


def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for 2x2*2x9.
    
    Partition 9 columns into 4 pairs + 1 single.
    Each pair uses 7 Strassen products, single uses 4 schoolbook products.
    Total rank: 4*7 + 4 = 32.
    """
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2."""
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # M1 = (a00 + a11) * (b00 + b11)
        c = start_col + 0
        U[u_00, c] += 1
        U[u_11, c] += 1
        V[v_00, c] += 1
        V[v_11, c] += 1
        W[w_00, c] += 1
        W[w_03, c] += 1
        
        # M2 = (a10 + a11) * b00
        c = start_col + 1
        U[u_10, c] += 1
        U[u_11, c] += 1
        V[v_00, c] += 1
        W[w_01, c] += 1
        W[w_03, c] -= 1
        
        # M3 = a00 * (b01 - b11)
        c = start_col + 2
        U[u_00, c] += 1
        V[v_01, c] += 1
        V[v_11, c] -= 1
        W[w_02, c] += 1
        W[w_03, c] += 1
        
        # M4 = a11 * (b10 - b00)
        c = start_col + 3
        U[u_11, c] += 1
        V[v_10, c] += 1
        V[v_00, c] -= 1
        W[w_00, c] += 1
        W[w_01, c] += 1
        
        # M5 = (a00 + a01) * b11
        c = start_col + 4
        U[u_00, c] += 1
        U[u_01, c] += 1
        V[v_11, c] += 1
        W[w_00, c] -= 1
        W[w_02, c] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        c = start_col + 5
        U[u_10, c] += 1
        U[u_00, c] -= 1
        V[v_00, c] += 1
        V[v_01, c] += 1
        W[w_03, c] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        c = start_col + 6
        U[u_01, c] += 1
        U[u_11, c] -= 1
        V[v_10, c] += 1
        V[v_11, c] += 1
        W[w_00, c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k."""
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        w_0 = k * n + 0
        w_1 = k * n + 1
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00, c] += 1
        V[v_0, c] += 1
        W[w_0, c] += 1
        
        c = start_col + 1
        U[u_01, c] += 1
        V[v_1, c] += 1
        W[w_0, c] += 1
        
        c = start_col + 2
        U[u_10, c] += 1
        V[v_0, c] += 1
        W[w_1, c] += 1
        
        c = start_col + 3
        U[u_11, c] += 1
        V[v_1, c] += 1
        W[w_1, c] += 1
        
        return start_col + 4
    
    # Process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    # Process single column: 8
    col = schoolbook_single(8, col)
    
    return U, V, W


def _winograd_2x2x3():
    """
    Attempt to construct a rank-10 decomposition for 2x2*2x3.
    
    The standard schoolbook approach gives rank 12 (4 per column * 3 columns).
    Strassen gives rank 7+4 = 11 (one Strassen pair + one schoolbook column).
    
    For rank 10, we need a non-trivial algorithm. Let me try to construct one
    using the Winograd F(4) algorithm adapted to this rectangular case.
    
    Winograd F(4) for 2x2*2x2: rank 7 (same as Strassen).
    
    For 2x2*2x3, let me try a different partition.
    
    Actually, let me just verify the Strassen+schoolbook approach (rank 11)
    and see if we can do better.
    
    For now, return the rank-11 approach (Strassen for 2 cols + schoolbook for 1 col).
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use Strassen for columns 0,1 and schoolbook for column 2
    col = 0
    
    # Strassen block for columns 0, 1
    k1, k2 = 0, 1
    v_00 = 0 * p + k1
    v_01 = 0 * p + k2
    v_10 = 1 * p + k1
    v_11 = 1 * p + k2
    
    w_00 = k1 * n + 0
    w_01 = k1 * n + 1
    w_02 = k2 * n + 0
    w_03 = k2 * n + 1
    
    u_00 = 0
    u_01 = 1
    u_10 = 2
    u_11 = 3
    
    # M1
    c = col
    U[u_00, c] += 1; U[u_11, c] += 1
    V[v_00, c] += 1; V[v_11, c] += 1
    W[w_00, c] += 1; W[w_03, c] += 1
    col += 1
    
    # M2
    c = col
    U[u_10, c] += 1; U[u_11, c] += 1
    V[v_00, c] += 1
    W[w_01, c] += 1; W[w_03, c] -= 1
    col += 1
    
    # M3
    c = col
    U[u_00, c] += 1
    V[v_01, c] += 1; V[v_11, c] -= 1
    W[w_02, c] += 1; W[w_03, c] += 1
    col += 1
    
    # M4
    c = col
    U[u_11, c] += 1
    V[v_10, c] += 1; V[v_00, c] -= 1
    W[w_00, c] += 1; W[w_01, c] += 1
    col += 1
    
    # M5
    c = col
    U[u_00, c] += 1; U[u_01, c] += 1
    V[v_11, c] += 1
    W[w_00, c] -= 1; W[w_02, c] += 1
    col += 1
    
    # M6
    c = col
    U[u_10, c] += 1; U[u_00, c] -= 1
    V[v_00, c] += 1; V[v_01, c] += 1
    W[w_03, c] += 1
    col += 1
    
    # M7
    c = col
    U[u_01, c] += 1; U[u_11, c] -= 1
    V[v_10, c] += 1; V[v_11, c] += 1
    W[w_00, c] += 1
    col += 1
    
    # Schoolbook for column 2
    k = 2
    v_0 = 0 * p + k
    v_1 = 1 * p + k
    w_0 = k * n + 0
    w_1 = k * n + 1
    
    c = col
    U[u_00, c] += 1; V[v_0, c] += 1; W[w_0, c] += 1
    col += 1
    
    c = col
    U[u_01, c] += 1; V[v_1, c] += 1; W[w_0, c] += 1
    col += 1
    
    c = col
    U[u_10, c] += 1; V[v_0, c] += 1; W[w_1, c] += 1
    col += 1
    
    c = col
    U[u_11, c] += 1; V[v_1, c] += 1; W[w_1, c] += 1
    col += 1
    
    return U, V, W


def _try_rank_30_decomposition(n, m, p):
    """
    Attempt to construct a rank-30 decomposition for 2x2*2x9.
    
    Strategy: Partition 9 columns into three groups of 3.
    For each group of 3 columns, use a rank-10 decomposition.
    Total rank: 3 * 10 = 30.
    
    The key is finding a valid rank-10 decomposition for 2x2*2x3.
    If we can't find one, this returns None.
    """
    # First, verify we have a valid rank-10 decomposition for 2x2*2x3
    U3, V3, W3 = _winograd_2x2x3()
    valid3, err3 = _verify_decomposition(U3, V3, W3, 2, 2, 3)
    rank3 = U3.shape[1]
    
    if not valid3 or rank3 > 10:
        return None
    
    # Now construct the rank-30 decomposition for 2x2*2x9
    # by placing three copies of the rank-10 decomposition
    rank = 3 * rank3
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(3):
        base = g * rank3
        cols = [g * 3 + c for c in range(3)]
        
        for r in range(rank3):
            # U is the same for each group
            U[:, base + r] = U3[:, r]
            
            # V: remap column indices
            for local_k in range(3):
                global_k = cols[local_k]
                for j in range(m):
                    V[j * p + global_k, base + r] = V3[j * 3 + local_k, r]
            
            # W: remap column indices
            for local_k in range(3):
                global_k = cols[local_k]
                for i in range(n):
                    W[global_k * n + i, base + r] = W3[local_k * 2 + i, r]
    
    return U, V, W


def _try_rank_28_decomposition(n, m, p):
    """
    Attempt to construct a rank-28 decomposition for 2x2*2x9.
    
    Strategy: Partition 9 columns into groups that use a more efficient algorithm.
    
    One approach: 2x2*2x8 using rank-24 + 2x2*2x1 using rank-4 = rank 28.
    For 2x2*2x8: partition into 4 pairs with Strassen = rank 28. Not enough.
    
    Alternative: Use a rank-24 algorithm for 2x2*2x8.
    2x2*2x4 has rank 12 (known). So 2x2*2x8 = 2 * rank(2x2*2x4) = 24.
    Then 2x2*2x9 = 24 + 4 = 28.
    
    Let me try to construct a rank-12 decomposition for 2x2*2x4.
    """
    # First, construct rank-12 for 2x2*2x4
    U4, V4, W4 = _strassen_2x2x4()
    if U4 is None:
        return None
    
    valid4, err4 = _verify_decomposition(U4, V4, W4, 2, 2, 4)
    if not valid4:
        return None
    
    rank4 = U4.shape[1]
    if rank4 > 12:
        return None
    
    # Construct rank-28 for 2x2*2x9
    # Two copies of rank-12 for columns 0-3 and 4-7, plus rank-4 for column 8
    rank = 2 * rank4 + 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(2):
        base = g * rank4
        cols = [g * 4 + c for c in range(4)]
        
        for r in range(rank4):
            U[:, base + r] = U4[:, r]
            
            for local_k in range(4):
                global_k = cols[local_k]
                for j in range(m):
                    V[j * p + global_k, base + r] = V4[j * 4 + local_k, r]
            
            for local_k in range(4):
                global_k = cols[local_k]
                for i in range(n):
                    W[global_k * n + i, base + r] = W4[local_k * 2 + i, r]
    
    # Schoolbook for column 8
    base = 2 * rank4
    k = 8
    u_00, u_01, u_10, u_11 = 0, 1, 2, 3
    v_0 = 0 * p + k
    v_1 = 1 * p + k
    w_0 = k * n + 0
    w_1 = k * n + 1
    
    c = base + 0
    U[u_00, c] += 1; V[v_0, c] += 1; W[w_0, c] += 1
    c = base + 1
    U[u_01, c] += 1; V[v_1, c] += 1; W[w_0, c] += 1
    c = base + 2
    U[u_10, c] += 1; V[v_0, c] += 1; W[w_1, c] += 1
    c = base + 3
    U[u_11, c] += 1; V[v_1, c] += 1; W[w_1, c] += 1
    
    return U, V, W


def _strassen_2x2x4():
    """
    Construct a rank-12 decomposition for 2x2*2x4.
    
    Use two Strassen blocks (one for cols 0,1 and one for cols 2,3).
    This gives rank 14, not 12.
    
    For rank 12, we need a more clever algorithm.
    Let me try a different approach: use the fact that 2x2*2x4 can be
    seen as computing C[:,0:4] = A * B[:,0:4].
    
    Actually, the known rank for 2x2*2x4 is 12 (Baur and Strassen).
    Let me implement it.
    
    The Baur-Strassen algorithm for 2x2*2x4:
    Uses 12 multiplications.
    
    For now, let me just use two Strassen blocks (rank 14) and see if
    we can optimize.
    """
    n, m, p = 2, 2, 4
    # Use two Strassen blocks: rank 14
    # This is not optimal but let's verify correctness first
    
    # Actually, let me try to construct rank 12 directly.
    # The key insight: for 2x2*2x4, we can use a tensor product structure.
    
    # Let me just return None for now and use the rank-14 approach
    # if needed, or find a better algorithm.
    
    # For simplicity, use two Strassen blocks (rank 14)
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3)]:
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # M1
        c = col
        U[u_00, c] += 1; U[u_11, c] += 1
        V[v_00, c] += 1; V[v_11, c] += 1
        W[w_00, c] += 1; W[w_03, c] += 1
        col += 1
        
        # M2
        c = col
        U[u_10, c] += 1; U[u_11, c] += 1
        V[v_00, c] += 1
        W[w_01, c] += 1; W[w_03, c] -= 1
        col += 1
        
        # M3
        c = col
        U[u_00, c] += 1
        V[v_01, c] += 1; V[v_11, c] -= 1
        W[w_02, c] += 1; W[w_03, c] += 1
        col += 1
        
        # M4
        c = col
        U[u_11, c] += 1
        V[v_10, c] += 1; V[v_00, c] -= 1
        W[w_00, c] += 1; W[w_01, c] += 1
        col += 1
        
        # M5
        c = col
        U[u_00, c] += 1; U[u_01, c] += 1
        V[v_11, c] += 1
        W[w_00, c] -= 1; W[w_02, c] += 1
        col += 1
        
        # M6
        c = col
        U[u_10, c] += 1; U[u_00, c] -= 1
        V[v_00, c] += 1; V[v_01, c] += 1
        W[w_03, c] += 1
        col += 1
        
        # M7
        c = col
        U[u_01, c] += 1; U[u_11, c] -= 1
        V[v_10, c] += 1; V[v_11, c] += 1
        W[w_00, c] += 1
        col += 1
    
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Tries multiple strategies in order of preference:
    1. Rank-28 (if a valid rank-12 2x2*2x4 decomposition is found)
    2. Rank-30 (if a valid rank-10 2x2*2x3 decomposition is found)
    3. Rank-32 (Strassen-block approach, always valid)
    
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Try rank-28
    try:
        result = _try_rank_28_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            valid, err = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        U, V, W = _strassen_block_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try rank-30
    try:
        result = _try_rank_30_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            valid, err = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        U, V, W = _strassen_block_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 3: Strassen-block (rank 32) - always works
    try:
        U, V, W = _strassen_block_decomposition(n, m, p)
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            rank = U.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback
    U, V, W = _strassen_block_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
