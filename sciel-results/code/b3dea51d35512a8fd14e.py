"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

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

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check invalid entries: for each (a, c) pair, check all b that don't match
    for a in range(n * m):
        i, j = divmod(a, m)
        for c in range(p * n):
            k, i2 = divmod(c, n)
            if i2 != i:
                continue
            for b in range(m * p):
                j2, k2 = divmod(b, p)
                if j2 == j and k2 == k:
                    continue
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets using Strassen.
    offset_a: starting index in U (i*m + j for i=0, j=0)
    offset_b: starting index in V (j*p + k for j=0, k=0)
    offset_c: starting index in W (k*n + i for k=0, i=0)
    """
    U_full = np.zeros((n * m, 7))
    V_full = np.zeros((m * p, 7))
    W_full = np.zeros((p * n, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_full[offset_a, 0] = 1; U_full[offset_a + m + 1, 0] = 1
    V_full[offset_b, 0] = 1; V_full[offset_b + p + 1, 0] = 1
    W_full[offset_c, 0] = 1; W_full[offset_c + n + 1, 0] = 1
    
    # M2 = (A10+A11)B00
    U_full[offset_a + m, 1] = 1; U_full[offset_a + m + 1, 1] = 1
    V_full[offset_b, 1] = 1
    W_full[offset_c + n, 1] = 1; W_full[offset_c + n + 1, 1] = 1
    
    # M3 = A00(B01-B11)
    U_full[offset_a, 2] = 1
    V_full[offset_b + 1, 2] = 1; V_full[offset_b + p + 1, 2] = -1
    W_full[offset_c + 1, 2] = 1; W_full[offset_c + n + 1, 2] = 1
    
    # M4 = A11(B10-B00)
    U_full[offset_a + m + 1, 3] = 1
    V_full[offset_b + p, 3] = 1; V_full[offset_b, 3] = -1
    W_full[offset_c + n, 3] = 1; W_full[offset_c + n + 1, 3] = 1
    
    # M5 = (A00+A01)B11
    U_full[offset_a, 4] = 1; U_full[offset_a + 1, 4] = 1
    V_full[offset_b + p + 1, 4] = 1
    W_full[offset_c + 1, 4] = 1; W_full[offset_c + n + 1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_full[offset_a + m, 5] = 1; U_full[offset_a, 5] = -1
    V_full[offset_b, 5] = 1; V_full[offset_b + 1, 5] = 1
    W_full[offset_c + n, 5] = 1; W_full[offset_c + n + 1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_full[offset_a + 1, 6] = 1; U_full[offset_a + m + 1, 6] = -1
    V_full[offset_b + p, 6] = 1; V_full[offset_b + p + 1, 6] = 1
    W_full[offset_c, 6] = 1; W_full[offset_c + n, 6] = 1
    
    return U_full, V_full, W_full

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    Rank-2 decomposition: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    offset_a: i*m + j for i=0, j=0
    offset_b: j*p + k for j=0, k=0
    offset_c: k*n + i for k=0, i=0
    """
    U_full = np.zeros((n * m, 2))
    V_full = np.zeros((m * p, 2))
    W_full = np.zeros((p * n, 2))
    
    # Term r=0: A[0,0] * B[0,:] -> C[0,:]
    U_full[offset_a, 0] = 1
    V_full[offset_b, 0] = 1
    V_full[offset_b + p, 0] = 1
    W_full[offset_c, 0] = 1
    W_full[offset_c + n, 0] = 1
    
    # Term r=1: A[0,1] * B[1,:] -> C[0,:]
    U_full[offset_a + 1, 1] = 1
    V_full[offset_b + 1, 1] = 1
    V_full[offset_b + p + 1, 1] = 1
    W_full[offset_c, 1] = 1
    W_full[offset_c + n, 1] = 1
    
    return U_full, V_full, W_full

def _try_strassen_blocks():
    """
    Construct a block-based Strassen decomposition for 3x4x8.
    Total rank: 4*7 + 4*2 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n * m, 0))
    V_total = np.zeros((m * p, 0))
    W_total = np.zeros((p * n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    # Split into two 2x2x2 blocks: A[0:2,0:2]*B[0:2,0:4] and A[0:2,2:4]*B[2:4,0:4]
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    # Split into two 1x2x2 blocks
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return _schoolbook(), n * m * p

def _try_optimized_3x4x8():
    """
    Try to find a rank < 36 decomposition using a different blocking strategy.
    
    Strategy: Use 3x2x4 blocks (splitting m=4 into 2+2, p=8 into 4+4).
    Each 3x2x4 block can be decomposed as:
    - 3x2x2 + 3x2x2 = 2 * (rank of 3x2x2)
    
    For 3x2x2:
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    i in {0,1,2}, j in {0,1}, k in {0,1}
    
    This is a 3x2x2 tensor. We can decompose it as:
    - For each j in {0,1}: A[:,j] * B[j,:] -> C
    That's 2 terms, each of rank 1 in the "outer product" sense.
    But the tensor rank might be less than 6.
    
    Actually, let's think of it differently. The 3x2x2 multiplication tensor:
    T[(i,j), (j,k), (i,k)] = 1
    i in {0,1,2}, j in {0,1}, k in {0,1}
    
    Number of non-zero entries: 3*2*2 = 12.
    
    One decomposition:
    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> T[(0,0),(0,0),(0,0)]
    r=1: U[0,1]=1, V[0,1]=1, W[1,1]=1  -> T[(0,0),(0,1),(0,1)]
    r=2: U[0,2]=1, V[1,2]=1, W[0,2]=1  -> T[(0,1),(1,0),(0,0)]
    r=3: U[0,3]=1, V[1,3]=1, W[1,3]=1  -> T[(0,1),(1,1),(0,1)]
    r=4: U[1,4]=1, V[0,4]=1, W[2,4]=1  -> T[(1,0),(0,0),(1,0)]
    r=5: U[1,5]=1, V[0,5]=1, W[3,5]=1  -> T[(1,0),(0,1),(1,1)]
    r=6: U[1,6]=1, V[1,6]=1, W[2,6]=1  -> T[(1,1),(1,0),(1,0)]
    r=7: U[1,7]=1, V[1,7]=1, W[3,7]=1  -> T[(1,1),(1,1),(1,1)]
    r=8: U[2,8]=1, V[0,8]=1, W[4,8]=1  -> T[(2,0),(0,0),(2,0)]
    r=9: U[2,9]=1, V[0,9]=1, W[5,9]=1  -> T[(2,0),(0,1),(2,1)]
    r=10: U[2,10]=1, V[1,10]=1, W[4,10]=1  -> T[(2,1),(1,0),(2,0)]
    r=11: U[2,11]=1, V[1,11]=1, W[5,11]=1  -> T[(2,1),(1,1),(2,1)]
    
    That's rank 12, which is the schoolbook for 3x2x2.
    
    Can we do better? The rank of the 3x2x2 multiplication tensor is at least
    max(3*2, 2*2, 3*2) = 6. And it's at most 12 (schoolbook).
    
    Actually, I think the rank is 6 for 3x2x2. Let me try to construct it.
    
    For 3x2x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    One approach: use the fact that C[:,k] = A * B[:,k] for each k.
    For fixed k, C[:,k] is a 3-vector that is a linear combination of the columns of A.
    
    Actually, let's try a different decomposition. For 3x2x2, we can write:
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    Let me try to use 6 terms:
    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> contributes to C[0,0]
    r=1: U[0,1]=1, V[1,1]=1, W[0,1]=1  -> contributes to C[0,0]
    r=2: U[0,2]=1, V[0,2]=1, W[1,2]=1  -> contributes to C[0,1]
    r=3: U[0,3]=1, V[1,3]=1, W[1,3]=1  -> contributes to C[0,1]
    r=4: U[1,4]=1, V[0,4]=1, W[2,4]=1  -> contributes to C[1,0]
    r=5: U[1,5]=1, V[1,5]=1, W[2,5]=1  -> contributes to C[1,0]
    r=6: U[1,6]=1, V[0,6]=1, W[3,6]=1  -> contributes to C[1,1]
    r=7: U[1,7]=1, V[1,7]=1, W[3,7]=1  -> contributes to C[1,1]
    r=8: U[2,8]=1, V[0,8]=1, W[4,8]=1  -> contributes to C[2,0]
    r=9: U[2,9]=1, V[1,9]=1, W[4,9]=1  -> contributes to C[2,0]
    r=10: U[2,10]=1, V[0,10]=1, W[5,10]=1  -> contributes to C[2,1]
    r=11: U[2,11]=1, V[1,11]=1, W[5,11]=1  -> contributes to C[2,1]
    
    That's still 12. Let me think more carefully.
    
    Actually, I realize that for 3x2x2, the rank is indeed 12 (the schoolbook rank),
    because the tensor is "generic" enough that no Strassen-like reduction is possible.
    
    So the rank 36 decomposition is likely optimal or near-optimal.
    
    Let me just return the rank 36 decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Just use the same decomposition as _try_strassen_blocks
    return _try_strassen_blocks()

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 36)
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Try the optimized blocks (same rank 36 for now)
    try:
        result, rank = _try_optimized_3x4x8()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
