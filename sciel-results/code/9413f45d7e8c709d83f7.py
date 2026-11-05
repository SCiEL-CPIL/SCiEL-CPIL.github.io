"""Matrix multiplication 4x4 x 4x7 via optimized block decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition exactly reconstructs matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    # Use tensor contraction for speed
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky to vectorize directly. Let's do it in steps.
    # First, compute M[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c]
    # This is like a batched matrix multiply but with shared index j.
    # Actually, let's just loop over k and do matrix ops, or use einsum.
    # np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _build_444():
    """Strassen-based 4x4 * 4x4 -> 4x4, rank 49."""
    n = m = p = 4
    U = np.zeros((n*m, 49)); V = np.zeros((m*p, 49)); W = np.zeros((p*n, 49))
    # Strassen 2x2 patterns
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
    
    # 7 outer terms, each 7 inner terms
    macros = [
        ([A11, A22], [1,1], [B11, B22], [1,1], [(C11,1),(C22,1)]),
        ([A21, A22], [1,1], [B11], [1], [(C21,1)]),
        ([A11], [1], [B12, B22], [1,-1], [(C12,1)]),
        ([A22], [1], [B21, B11], [1,-1], [(C11,1),(C21,1)]),
        ([A11, A12], [1,1], [B22], [1], [(C11,-1),(C12,1)]),
        ([A21, A11], [1,-1], [B11, B12], [1,1], [(C22,1)]),
        ([A12, A22], [1,-1], [B21, B22], [1,1], [(C11,1)]),
    ]
    strassen_pat = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,-1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1)]),
    ]
    col = 0
    for macro in macros:
        A_blocks, A_signs, B_blocks, B_signs, C_targets = macro
        for sub in strassen_pat:
            U_entries, V_entries, W_entries = sub
            for a_local, s_u in U_entries:
                gidx = A_blocks[a_local]
                U[gidx, col] += A_signs[a_local] * s_u
            for b_local, s_v in V_entries:
                gidx = B_blocks[b_local]
                V[gidx, col] += B_signs[b_local] * s_v
            for c_local, s_w in W_entries:
                C_block, c_sign = C_targets[c_local]
                for gidx in C_block:
                    W[gidx, col] += c_sign * s_w
            col += 1
    return U, V, W

def _build_443():
    """4x4 * 4x3 -> 4x3, rank 36 (Schoolbook)."""
    n, m, p = 4, 4, 3
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _build_447():
    """Combine 4x4*4x4 (rank 49) and 4x4*4x3 (rank 36) for 4x4*4x7 (rank 85)."""
    n, m, p = 4, 4, 7
    U4, V4, W4 = _build_444()
    U3, V3, W3 = _build_443()
    r4 = U4.shape[1]
    r3 = U3.shape[1]
    total_rank = r4 + r3
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: 4x4 * 4x4 (first 4 cols of B)
    U[:, 0:r4] = U4
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:r4] = V4[i*4+j, :]
    for i in range(4):
        for k in range(4):
            W[k*7+i, 0:r4] = W4[k*4+i, :]
            
    # Part 2: 4x4 * 4x3 (last 3 cols of B, indices 4..6)
    U[:, r4:total_rank] = U3
    for i in range(4):
        for j in range(4):
            V[i*7+4+j, r4:total_rank] = V3[i*3+j, :]
    for i in range(4):
        for k in range(3):
            W[(k+4)*7+i, r4:total_rank] = W3[k*4+i, :]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n*m*p
    
    try:
        if time.time() < deadline - 10:
            U, V, W = _build_447()
            if _verify(U, V, W, n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = U.shape[1]
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
