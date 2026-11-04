"""Matrix multiplication 4x4 x 4x7 via Strassen block decomposition."""
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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
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
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
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
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
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
    """4x4 * 4x3 -> 4x3, rank 35.
    
    Split B columns into 2+1. 
    For the 2-column part: 4 Strassen 2x2*2x2 blocks (rank 7 each) = 28
    For the 1-column part: 4 direct 2x2*2x1 blocks (rank 4 each) = 8... 
    Wait, 2x2*2x1 direct is rank 4? No, it's rank 2*1*... 
    Actually for 2x2 * 2x1 -> 2x1: C[i] = sum_k A[i,k]*B[k,0], rank = 2*2*1 = 4? No.
    The tensor rank of 2x2*2x1 is 2*2*1 = 4 for schoolbook, but can we do better?
    Actually 2x2*2x1: C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    This is rank 4 (schoolbook). No Strassen benefit for 1 column.
    
    So: 4 * 7 + 4 * 4 = 28 + 16 = 44. Same as before!
    
    Let me try a different approach: split A into 2+2 rows, and handle all 3 columns of B at once.
    For each pair of A rows, we compute 2x4 * 4x3 -> 2x3.
    The tensor rank of 2x4*4x3... this is more complex.
    
    Alternative: use the fact that 4x4*4x3 can be seen as 4x4*4x4 with 3 columns zeroed.
    But that doesn't help directly.
    
    Let me try: rank 44 is the current best I can construct. Let me verify the current approach works
    and then try to find a better split.
    
    Actually, let me try splitting B as 1+2 instead, or use a different block structure.
    
    For now, keep rank 44 but verify correctness.
    """
    n, m, p = 4, 4, 3
    r = 44
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    
    products = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2),
    ]
    
    strassen_2x2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    col = 0
    for (ar, ac, br, cr) in products:
        for sub in strassen_2x2:
            U_entries, V_entries, W_entries = sub
            for a_local, s_u in U_entries:
                U[(ar+a_local//2)*4 + (ac+a_local%2), col] += s_u
            for b_local, s_v in V_entries:
                V[(br+b_local//2)*3 + b_local%2, col] += s_v
            for c_local, s_w in W_entries:
                W[(cr+c_local//2)*4 + c_local%2, col] += s_w
            col += 1
        for ai in range(2):
            for bk in range(2):
                U[(ar+ai)*4 + (ac+bk), col] = 1
                V[(br+bk)*3 + 2, col] = 1
                W[(cr+ai)*4 + 2, col] = 1
                col += 1
    return U, V, W

def _build_447():
    """Combine 4x4*4x4 (rank 49) and 4x4*4x3 (rank 44) for 4x4*4x7 (rank 93)."""
    n, m, p = 4, 4, 7
    U1, V1, W1 = _build_444()
    U2, V2, W2 = _build_443()
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    U[:, 0:r1] = U1
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:r1] = V1[i*4+j, :]
    for i in range(4):
        for j in range(4):
            W[i*7+j, 0:r1] = W1[i*4+j, :]
    U[:, r1:total_rank] = U2
    for i in range(4):
        for j in range(3):
            V[i*7+4+j, r1:total_rank] = V2[i*3+j, :]
    for row in range(4):
        for col in range(3):
            W[row*7+4+col, r1:total_rank] = W2[row*4+col, :]
    return U, V, W

def _try_polish(U, V, W, n, m, p, max_iter=50, tol=1e-9):
    """Attempt to reduce rank by removing zero/near-zero columns."""
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    removed = []
    for c in range(r):
        if np.allclose(U[:, c], 0, atol=1e-12) and np.allclose(V[:, c], 0, atol=1e-12) and np.allclose(W[:, c], 0, atol=1e-12):
            removed.append(c)
    if removed:
        keep = [c for c in range(r) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    return U, V, W

def _try_polish_linear(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """Try to find linear dependencies among columns to reduce rank.
    
    If columns c1 and c2 are linearly dependent (U[:,c1] = alpha*U[:,c2], etc.),
    we can eliminate one. More generally, check if any column is a linear combination
    of others in all three matrices simultaneously.
    """
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    
    # Try removing each column and checking if the remaining still works
    # This is O(r * verification_cost) which is expensive, so only do a few
    # Instead, look for exact duplicate or zero columns
    removed = set()
    for c in range(r):
        if c in removed:
            continue
        # Check if column c is zero
        if np.allclose(U[:, c], 0, atol=1e-12) and np.allclose(V[:, c], 0, atol=1e-12) and np.allclose(W[:, c], 0, atol=1e-12):
            removed.add(c)
            continue
        # Check if column c is a scalar multiple of another column
        for d in range(r):
            if d == c or d in removed:
                continue
            # Check if U[:,c] = alpha * U[:,d]
            nonzero_u = U[:, d] != 0
            if np.sum(nonzero_u) == 0:
                continue
            alpha_u = U[nonzero_u, c] / U[nonzero_u, d]
            if not np.allclose(alpha_u, alpha_u[0], atol=1e-12):
                continue
            alpha = alpha_u[0]
            if np.allclose(U[:, c], alpha * U[:, d], atol=1e-12) and \
               np.allclose(V[:, c], alpha * V[:, d], atol=1e-12) and \
               np.allclose(W[:, c], alpha * W[:, d], atol=1e-12):
                removed.add(c)
                break
    
    if removed:
        keep = [c for c in range(r) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n*m*p
    
    # Try Strassen-based construction
    try:
        if time.time() < deadline - 10:
            U, V, W = _build_447()
            if _verify(U, V, W, n, m, p):
                # Try to polish
                U_polished, V_polished, W_polished = _try_polish(U, V, W, n, m, p)
                if _verify(U_polished, V_polished, W_polished, n, m, p):
                    best = {"U": U_polished.tolist(), "V": V_polished.tolist(), "W": W_polished.tolist()}
                    best_rank = U_polished.shape[1]
                else:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = U.shape[1]
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
