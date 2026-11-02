"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _verify(U, V, W):
    """Verify that the decomposition is correct - checks ALL entries."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # and check it matches the matrix multiplication tensor
    
    # Valid entries: T[i*m+j, j*p+k, k*n+i] = 1
    # Invalid entries: all other (a,b,c) should be 0
    
    # Efficient check: for each rank element r, add outer product contribution
    # But for small sizes, just compute directly
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :]) @ W[c, :]
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check all invalid entries (comprehensive)
    # Invalid: (a, b, c) where not (b - j*p == k and c - k*n == i and a - i*m == j)
    # i.e., the indices don't form a valid (i,j,k) triple
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a valid entry
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    continue  # valid entry, already checked
                
                val = np.dot(U[a, :], V[b, :]) @ W[c, :]
                if abs(val) > 1e-6:
                    return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets.
    offset_a: starting index in A (i0*m + j0)
    offset_b: starting index in B (j0*p + k0)
    offset_c: starting index in C (k0*n + i0)
    
    Local indices within the 2x2 block:
    A: A00=offset_a+0, A01=offset_a+1, A10=offset_a+m, A11=offset_a+m+1
    B: B00=offset_b+0, B01=offset_b+1, B10=offset_b+p, B11=offset_b+p+1
    C: C00=offset_c+0, C01=offset_c+1, C10=offset_c+n, C11=offset_c+n+1
    """
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # Local index offsets
    a00, a01, a10, a11 = offset_a, offset_a+1, offset_a+m, offset_a+m+1
    b00, b01, b10, b11 = offset_b, offset_b+1, offset_b+p, offset_b+p+1
    c00, c01, c10, c11 = offset_c, offset_c+1, offset_c+n, offset_c+n+1
    
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    U[a00, 0] = 1
    U[a11, 0] = 1
    V[b00, 0] = 1
    V[b11, 0] = 1
    W[c00, 0] = 1
    W[c11, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 + C11
    U[a10, 1] = 1
    U[a11, 1] = 1
    V[b00, 1] = 1
    W[c10, 1] = 1
    W[c11, 1] = 1
    
    # M3 = A00(B01-B11) -> C01 + C11
    U[a00, 2] = 1
    V[b01, 2] = 1
    V[b11, 2] = -1
    W[c01, 2] = 1
    W[c11, 2] = 1
    
    # M4 = A11(B10-B00) -> C10 + C11
    U[a11, 3] = 1
    V[b10, 3] = 1
    V[b00, 3] = -1
    W[c10, 3] = 1
    W[c11, 3] = 1
    
    # M5 = (A00+A01)B11 -> C01 + C11
    U[a00, 4] = 1
    U[a01, 4] = 1
    V[b11, 4] = 1
    W[c01, 4] = 1
    W[c11, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C10 + C11
    U[a10, 5] = 1
    U[a00, 5] = -1
    V[b00, 5] = 1
    V[b01, 5] = 1
    W[c10, 5] = 1
    W[c11, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 + C11
    U[a01, 6] = 1
    U[a11, 6] = -1
    V[b10, 6] = 1
    V[b11, 6] = 1
    W[c00, 6] = 1
    W[c11, 6] = 1
    
    return U, V, W

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    offset_a: starting index in A (i0*m + j0)
    offset_b: starting index in B (j0*p + k0)
    offset_c: starting index in C (k0*n + i0)
    
    Local indices:
    A: A0=offset_a+0, A1=offset_a+1
    B: B00=offset_b+0, B01=offset_b+1, B10=offset_b+p, B11=offset_b+p+1
    C: C0=offset_c+0, C1=offset_c+1
    """
    U = np.zeros((n*m, 2))
    V = np.zeros((m*p, 2))
    W = np.zeros((p*n, 2))
    
    # r=0: A0 * (B00 + B01) -> C0 + C1
    U[offset_a, 0] = 1
    V[offset_b, 0] = 1
    V[offset_b + 1, 0] = 1
    W[offset_c, 0] = 1
    W[offset_c + 1, 0] = 1
    
    # r=1: A1 * (B10 + B11) -> C0 + C1
    U[offset_a + 1, 1] = 1
    V[offset_b + p, 1] = 1
    V[offset_b + p + 1, 1] = 1
    W[offset_c, 1] = 1
    W[offset_c + 1, 1] = 1
    
    return U, V, W

def _try_strassen_blocks():
    """
    Try to use Strassen-like block structure for 3x4x8.
    Multiple tiling strategies, return the best (lowest rank) verified one.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    def _add_block(U_total, V_total, W_total, U_block, V_block, W_block):
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
        return U_total, V_total, W_total
    
    def _try_tiling(splits):
        """
        splits: list of (i_start, i_end, j_start, j_end, k_start, k_end, algo)
        algo: 'strassen' for 2x2x2, 'dot' for 1x2x2, 'schoolbook' for 1x1x1
        """
        U_total = np.zeros((n*m, 0))
        V_total = np.zeros((m*p, 0))
        W_total = np.zeros((p*n, 0))
        
        for (i0, i1, j0, j1, k0, k1, algo) in splits:
            offset_a = i0 * m + j0
            offset_b = j0 * p + k0
            offset_c = k0 * n + i0
            
            if algo == 'strassen':
                U_b, V_b, W_b = _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p)
            elif algo == 'dot':
                U_b, V_b, W_b = _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p)
            else:
                # schoolbook 1x1x1
                U_b = np.zeros((n*m, 1))
                V_b = np.zeros((m*p, 1))
                W_b = np.zeros((p*n, 1))
                U_b[offset_a, 0] = 1
                V_b[offset_b, 0] = 1
                W_b[offset_c, 0] = 1
            
            U_total, V_total, W_total = _add_block(U_total, V_total, W_total, U_b, V_b, W_b)
        
        U_list = U_total.tolist()
        V_list = V_total.tolist()
        W_list = W_total.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}, U_total.shape[1]
        return None, float('inf')
    
    # Tiling 1: (2+1) x (2+2) x (2+2+2+2)
    # For each (i-block, j-block, k-block), use appropriate algorithm
    # i-blocks: [0:2] (size 2), [2:3] (size 1)
    # j-blocks: [0:2] (size 2), [2:4] (size 2)
    # k-blocks: [0:2], [2:4], [4:6], [6:8] (all size 2)
    
    splits1 = []
    for i0, i1 in [(0, 2), (2, 3)]:
        for j0, j1 in [(0, 2), (2, 4)]:
            for k0, k1 in [(0, 2), (2, 4), (4, 6), (6, 8)]:
                if (i1-i0) == 2 and (j1-j0) == 2 and (k1-k0) == 2:
                    algo = 'strassen'
                elif (i1-i0) == 1 and (j1-j0) == 2 and (k1-k0) == 2:
                    algo = 'dot'
                else:
                    algo = 'schoolbook'
                splits1.append((i0, i1, j0, j1, k0, k1, algo))
    
    result1, rank1 = _try_tiling(splits1)
    if result1 and rank1 < best_rank:
        best_result = result1
        best_rank = rank1
    
    # Tiling 2: (2+1) x 4 x (4+4) - split k into two 4s
    # 2x4x4 blocks: can split j into (2+2), giving 2x2x4 blocks
    # 2x2x4: split k into (2+2), giving 2x2x2 blocks (Strassen)
    # 1x4x4 blocks: split j into (2+2), giving 1x2x4 blocks
    # 1x2x4: split k into (2+2), giving 1x2x2 blocks (dot)
    
    splits2 = []
    for i0, i1 in [(0, 2), (2, 3)]:
        for j0, j1 in [(0, 2), (2, 4)]:
            for k0, k1 in [(0, 4), (4, 8)]:
                # For 2x2x4: use Strassen on two 2x2x2
                if (i1-i0) == 2 and (j1-j0) == 2 and (k1-k0) == 4:
                    # Split k into two 2s
                    for kk0, kk1 in [(k0, k0+2), (k0+2, k0+4)]:
                        splits2.append((i0, i1, j0, j1, kk0, kk1, 'strassen'))
                elif (i1-i0) == 1 and (j1-j0) == 2 and (k1-k0) == 4:
                    # Split k into two 2s
                    for kk0, kk1 in [(k0, k0+2), (k0+2, k0+4)]:
                        splits2.append((i0, i1, j0, j1, kk0, kk1, 'dot'))
    
    result2, rank2 = _try_tiling(splits2)
    if result2 and rank2 < best_rank:
        best_result = result2
        best_rank = rank2
    
    # Tiling 3: (1+1+1) x 4 x 8 - all 1x4x8, use dot for 1x2x2
    splits3 = []
    for i0, i1 in [(0, 1), (1, 2), (2, 3)]:
        for j0, j1 in [(0, 2), (2, 4)]:
            for k0, k1 in [(0, 2), (2, 4), (4, 6), (6, 8)]:
                splits3.append((i0, i1, j0, j1, k0, k1, 'dot'))
    
    result3, rank3 = _try_tiling(splits3)
    if result3 and rank3 < best_rank:
        best_result = result3
        best_rank = rank3
    
    # Tiling 4: (2+1) x (2+2) x 8 with mixed approaches
    # Try: 2x2x8 as two Strassen 2x2x4, each as two Strassen 2x2x2
    # 1x2x8 as four dot 1x2x2
    splits4 = []
    for i0, i1 in [(0, 2), (2, 3)]:
        for j0, j1 in [(0, 2), (2, 4)]:
            for k0, k1 in [(0, 2), (2, 4), (4, 6), (6, 8)]:
                if (i1-i0) == 2:
                    splits4.append((i0, i1, j0, j1, k0, k1, 'strassen'))
                else:
                    splits4.append((i0, i1, j0, j1, k0, k1, 'dot'))
    
    result4, rank4 = _try_tiling(splits4)
    if result4 and rank4 < best_rank:
        best_result = result4
        best_rank = rank4
    
    if best_result is not None:
        return best_result, best_rank
    else:
        return _schoolbook(), n * m * p

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
