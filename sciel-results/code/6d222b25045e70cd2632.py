"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_decomposition(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, 
                         n_total, m_total, p_total, 
                         offset_a, offset_b, offset_c):
    """
    Embed a sub-decomposition into the global factor matrices.
    """
    rank = U_sub.shape[1]
    U_global = np.zeros((n_total * m_total, rank), dtype=np.int64)
    V_global = np.zeros((m_total * p_total, rank), dtype=np.int64)
    W_global = np.zeros((p_total * n_total, rank), dtype=np.int64)
    
    for a_sub in range(n_sub):
        for b_sub in range(m_sub):
            idx_u_sub = a_sub * m_sub + b_sub
            idx_u_global = (offset_a + a_sub) * m_total + (offset_b + b_sub)
            U_global[idx_u_global] = U_sub[idx_u_sub]
            
    for b_sub in range(m_sub):
        for c_sub in range(p_sub):
            idx_v_sub = b_sub * p_sub + c_sub
            idx_v_global = (offset_b + b_sub) * p_total + (offset_c + c_sub)
            V_global[idx_v_global] = V_sub[idx_v_sub]
            
    for c_sub in range(p_sub):
        for a_sub in range(n_sub):
            idx_w_sub = c_sub * n_sub + a_sub
            idx_w_global = (offset_c + c_sub) * n_total + (offset_a + a_sub)
            W_global[idx_w_global] = W_sub[idx_w_sub]
            
    return U_global, V_global, W_global

def _compose_decompositions(decomps, n, m, p):
    """
    Compose a list of (U, V, W) decompositions that are block-aligned.
    """
    if not decomps:
        return np.zeros((n*m, 0)), np.zeros((m*p, 0)), np.zeros((p*n, 0))
    
    U_list = []
    V_list = []
    W_list = []
    
    for U, V, W in decomps:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    U_total = np.hstack(U_list)
    V_total = np.hstack(V_list)
    W_total = np.hstack(W_list)
    
    return U_total, V_total, W_total

def _build_5x5_decomposition():
    """
    Construct 5x5x5 decomposition using 1+4 split.
    Total Rank: 81.
    """
    n, m, p = 5, 5, 5
    
    # Split 5 into 1 and 4
    n1, n2 = 1, 4
    m1, m2 = 1, 4
    p1, p2 = 1, 4
    
    decomps = []
    
    # Block (0,0): A[0:1, 0:1], B[0:1, 0:1], C[0:1, 0:1]
    # Dims: 1x1x1
    U00, V00, W00 = _schoolbook(1, 1, 1)
    U00, V00, W00 = _embed_decomposition(U00, V00, W00, 1, 1, 1, n, m, p, 0, 0, 0)
    decomps.append((U00, V00, W00))
    
    # Block (0,1): A[0:1, 0:1], B[0:1, 1:5], C[1:5, 0:1]
    # Dims: 1x1x4
    U01, V01, W01 = _schoolbook(1, 1, 4)
    U01, V01, W01 = _embed_decomposition(U01, V01, W01, 1, 1, 4, n, m, p, 0, 0, 1)
    decomps.append((U01, V01, W01))
    
    # Block (1,0): A[1:5, 0:1], B[0:1, 0:1], C[0:1, 1:5]
    # Dims: 4x1x1
    U10, V10, W10 = _schoolbook(4, 1, 1)
    U10, V10, W10 = _embed_decomposition(U10, V10, W10, 4, 1, 1, n, m, p, 1, 0, 0)
    decomps.append((U10, V10, W10))
    
    # Block (1,1): A[1:5, 1:5], B[1:5, 1:5], C[1:5, 1:5]
    # Dims: 4x4x4
    # Use Strassen for 4x4x4
    # Construct Strassen 4x4x4 from 2x2x2 blocks
    # Split 4 into 2+2
    U44_sub = []
    V44_sub = []
    W44_sub = []
    
    # 4x4x4 = 4 blocks of 2x2x2
    # (0,0) 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 0, 0, 0)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (0,1) 2x2x2 (A[0:2, 0:2], B[0:2, 2:4], C[2:4, 0:2])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 0, 0, 2)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (1,0) 2x2x2 (A[2:4, 0:2], B[0:2, 0:2], C[0:2, 2:4])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 2, 0, 0)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    # (1,1) 2x2x2 (A[2:4, 2:4], B[2:4, 2:4], C[2:4, 2:4])
    U_s, V_s, W_s = _strassen_2x2()
    U_s, V_s, W_s = _embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 4, 4, 4, 2, 2, 2)
    U44_sub.append(U_s); V44_sub.append(V_s); W44_sub.append(W_s)
    
    U44, V44, W44 = _compose_decompositions(list(zip(U44_sub, V44_sub, W44_sub)), 4, 4, 4)
    
    # Embed 4x4x4 into 5x5x5 at offset (1,1,1)
    U11, V11, W11 = _embed_decomposition(U44, V44, W44, 4, 4, 4, n, m, p, 1, 1, 1)
    decomps.append((U11, V11, W11))
    
    # Cross terms
    # C[0:1, 0:1] = A[0:1, 0:1]B[0:1, 0:1] + A[0:1, 1:5]B[1:5, 0:1]
    # Second term: A[0:1, 1:5] (1x4), B[1:5, 0:1] (4x1) -> C[0:1, 0:1] (1x1).
    # Dims: 1x4x1. Rank 4.
    U01_cross, V01_cross, W01_cross = _schoolbook(1, 4, 1)
    U01_cross, V01_cross, W01_cross = _embed_decomposition(U01_cross, V01_cross, W01_cross, 1, 4, 1, n, m, p, 0, 1, 0)
    decomps.append((U01_cross, V01_cross, W01_cross))
    
    # C[0:1, 1:5] = A[0:1, 0:1]B[0:1, 1:5] + A[0:1, 1:5]B[1:5, 1:5]
    # Second term: 1x4x4. Rank 4.
    U01_main, V01_main, W01_main = _schoolbook(1, 4, 4)
    U01_main, V01_main, W01_main = _embed_decomposition(U01_main, V01_main, W01_main, 1, 4, 4, n, m, p, 0, 1, 1)
    decomps.append((U01_main, V01_main, W01_main))
    
    # C[1:5, 0:1] = A[1:5, 0:1]B[0:1, 0:1] + A[1:5, 1:5]B[1:5, 0:1]
    # Second term: 4x4x1. Rank 4.
    U10_main, V10_main, W10_main = _schoolbook(4, 4, 1)
    U10_main, V10_main, W10_main = _embed_decomposition(U10_main, V10_main, W10_main, 4, 4, 1, n, m, p, 1, 1, 0)
    decomps.append((U10_main, V10_main, W10_main))
    
    # C[1:5, 1:5] = A[1:5, 0:1]B[0:1, 1:5] + A[1:5, 1:5]B[1:5, 1:5]
    # First term: 4x1x4. Rank 4.
    U11_cross, V11_cross, W11_cross = _schoolbook(4, 1, 4)
    U11_cross, V11_cross, W11_cross = _embed_decomposition(U11_cross, V11_cross, W11_cross, 4, 1, 4, n, m, p, 1, 0, 1)
    decomps.append((U11_cross, V11_cross, W11_cross))
    
    U_total, V_total, W_total = _compose_decompositions(decomps, n, m, p)
    
    return U_total.tolist(), V_total.tolist(), W_total.tolist()

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Generate the structured decomposition
    U, V, W = _build_5x5_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
