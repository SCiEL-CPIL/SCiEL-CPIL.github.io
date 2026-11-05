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
    # Strassen 2x2 patterns
    # A11=[0,1,4,5], A12=[2,3,6,7], A21=[8,9,12,13], A22=[10,11,14,15]
    # B11=[0,1,4,5], B12=[2,3,6,7], B21=[8,9,12,13], B22=[10,11,14,15]
    # C11=[0,1,4,5], C12=[2,3,6,7], C21=[8,9,12,13], C22=[10,11,14,15]
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
    
    # 7 outer terms, each 7 inner terms
    # Outer terms: (A_blocks, A_signs, B_blocks, B_signs, C_targets)
    # C_targets: list of (C_block, sign)
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

def _build_442():
    """4x4 * 4x2 -> 4x2, rank 16 (Schoolbook)."""
    n, m, p = 4, 4, 2
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _build_441():
    """4x4 * 4x1 -> 4x1, rank 16 (Schoolbook)."""
    n, m, p = 4, 4, 1
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _build_445():
    """4x4 * 4x5 -> 4x5, rank 65 (Strassen 49 + Schoolbook 16)."""
    n, m, p = 4, 4, 5
    U444, V444, W444 = _build_444()
    U441, V441, W441 = _build_441()
    r1 = U444.shape[1]
    r2 = U441.shape[1]
    total_rank = r1 + r2
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: 4x4 * 4x4 (first 4 cols of B)
    U[:, 0:r1] = U444
    for i in range(4):
        for j in range(4):
            V[i*5+j, 0:r1] = V444[i*4+j, :]
            W[i*5+j, 0:r1] = W444[i*4+j, :] # Wait, W is p*n. p=5, n=4.
            # W444 is 16x49. W is 20x65.
            # W444 index k*n+i. k in 0..3.
            # W index k*5+i? No, W[k*n+i] is for 4x4*4x4.
            # For 4x4*4x5, W index is k*4+i? No, W is (p*n, rank).
            # W[k*n + i] where k is col of C (0..p-1), i is row of C (0..n-1).
            # So W index is k*n + i.
            # In 444, k in 0..3. In 445, k in 0..3 for the 444 part.
            # So W[k*4+i, 0:r1] = W444[k*4+i, :]
            pass # Handled below
    
    # Correct mapping for 444 part in 445
    for i in range(4):
        for k in range(4):
            W[k*4+i, 0:r1] = W444[k*4+i, :]
            
    # Part 2: 4x4 * 4x1 (last 1 col of B, index 4)
    U[:, r1:total_rank] = U441
    for i in range(4):
        for j in range(4):
            V[i*5+4, r1:total_rank] = V441[i*1+j, :] # V441 is 4x16. V441[j, c]
            # V441 index j*p+k = j*1+0 = j.
            # V445 index i*5+4.
            # V441[i,j] corresponds to A[i,j] * B[j,0].
            # So V441[j, c] goes to V445[i*5+4, r1+c]? No.
            # V441 is (m*p, rank) = (4*1, 16) = (4, 16).
            # V441[j, c] is 1 if term c uses B[j,0].
            # In 445, B[j,4] is the 5th column.
            # So V445[j*5+4, r1+c] = V441[j, c].
            pass
            
    for j in range(4):
        for c in range(r2):
            V[j*5+4, r1+c] = V441[j, c]
            
    for i in range(4):
        for k in range(4): # k is row of C? No, k is col of C in W[k*n+i]
            # W441 is (p*n, rank) = (1*4, 16) = (4, 16).
            # W441[k*1+i, c] = W441[k+i, c]? No, W441 index is k*n+i = k*4+i.
            # But p=1. So W441 index is k*1+i? No, n=4.
            # W441 shape (4, 16). Index k*n+i? k=0. i=0..3.
            # So W441[i, c].
            # In 445, W index for col 4 is k*4+i where k=4.
            # So W[4*4+i, r1+c] = W441[i, c].
            pass
            
    for i in range(4):
        for c in range(r2):
            W[4*4+i, r1+c] = W441[i, c]
            
    return U, V, W

def _build_447():
    """Combine 4x4*4x2 (rank 16) and 4x4*4x5 (rank 65) for 4x4*4x7 (rank 81)."""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _build_442()
    U5, V5, W5 = _build_445()
    r2 = U2.shape[1]
    r5 = U5.shape[1]
    total_rank = r2 + r5
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: 4x4 * 4x2 (first 2 cols of B)
    U[:, 0:r2] = U2
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:r2] = V2[i*2+j, :]
    for i in range(4):
        for k in range(2):
            W[k*7+i, 0:r2] = W2[k*2+i, :] # W2 is (2*4, 16). Index k*2+i? No.
            # W2 index is k*n+i = k*4+i.
            # W2 shape (8, 16).
            # W2[k*4+i, c].
            # W index is k*7+i.
            pass
            
    for i in range(4):
        for k in range(2):
            W[k*7+i, 0:r2] = W2[k*4+i, :]

    # Part 2: 4x4 * 4x5 (last 5 cols of B, indices 2..6)
    U[:, r2:total_rank] = U5
    for i in range(4):
        for j in range(4):
            V[i*7+2+j, r2:total_rank] = V5[i*5+j, :]
    for i in range(4):
        for k in range(5):
            W[k*7+i, r2:total_rank] = W5[k*5+i, :]
            
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
