"""
Matrix multiplication 2,2,15 tensor rank minimization.
Uses blockwise Strassen decomposition with a structured search for
lower-rank 2x2x3 blocks.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute sum_k U[:,k] * V[:,k] * W[:,k]^T
    # Using einsum: first contract U and V over k
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11-A22)(B21+B22)
    U[0,0]=1; U[3,0]=-1; V[1,0]=1; V[3,0]=1; W[2,0]=1; W[3,0]=1
    # M2 = (A21+A22)B11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    # M3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=-1
    # M4 = A22(B21-B11)
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=-1
    # M5 = (A11+A22)(B22-B12)
    U[0,4]=1; U[3,4]=1; V[3,4]=1; V[1,4]=-1; W[0,4]=1; W[3,4]=-1
    # M6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1; W[1,5]=1
    # M7 = (A22-A21)(B12+B22)
    U[3,6]=1; U[2,6]=-1; V[1,6]=1; V[3,6]=1; W[1,6]=1; W[3,6]=1
    return U, V, W

def _rank3_221():
    """Rank-3 decomposition for 2x2x1 matrix multiplication."""
    U = np.zeros((4, 3))
    V = np.zeros((2, 3))
    W = np.zeros((2, 3))
    U[:,0]=[1,0,0,0]; V[:,0]=[1,0]; W[:,0]=[1,0]
    U[:,1]=[0,0,1,0]; V[:,1]=[0,1]; W[:,1]=[0,1]
    U[:,2]=[0,1,0,1]; V[:,2]=[1,1]; W[:,2]=[1,1]
    return U, V, W

def _build_223_rank10():
    """
    Build 2x2x3 as concatenation of 2x2x2 (Strassen) and 2x2x1 (rank 3).
    Total rank = 7 + 3 = 10.
    U: (4, 10), V: (6, 10), W: (6, 10)
    V indices: j*3+k, W indices: k*2+i
    """
    U_s, V_s, W_s = _strassen_222()
    U_r, V_r, W_r = _rank3_221()
    
    U = np.zeros((4, 10))
    V = np.zeros((6, 10))
    W = np.zeros((6, 10))
    
    # First 7 columns: 2x2x2 for k=0,1
    U[:, :7] = U_s
    for j in range(2):
        for kl in range(2):
            v_loc = j * 2 + kl
            v_glob = j * 3 + kl
            V[v_glob, :7] = V_s[v_loc]
    for kl in range(2):
        for i in range(2):
            w_loc = kl * 2 + i
            w_glob = kl * 2 + i
            W[w_glob, :7] = W_s[w_loc]
    
    # Last 3 columns: 2x2x1 for k=2
    U[:, 7:] = U_r
    for j in range(2):
        v_glob = j * 3 + 2
        V[v_glob, 7:] = V_r[j]
    for i in range(2):
        w_glob = 2 * 2 + i
        W[w_glob, 7:] = W_r[i]
    
    return U, V, W

def _schoolbook(n, m, p):
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

def _build_2215_blockwise():
    """5 blocks of 2x2x3: rank = 5 * 10 = 50."""
    n, m, p = 2, 2, 15
    U_223, V_223, W_223 = _build_223_rank10()
    r223 = U_223.shape[1]
    
    U = np.zeros((n * m, 5 * r223))
    V = np.zeros((m * p, 5 * r223))
    W = np.zeros((p * n, 5 * r223))
    
    for block in range(5):
        k0 = block * 3
        col = block * r223
        U[:, col:col+r223] = U_223
        for j in range(m):
            for kl in range(3):
                v_loc = j * 3 + kl
                v_glob = j * p + (k0 + kl)
                V[v_glob, col:col+r223] = V_223[v_loc]
        for kl in range(3):
            for i in range(n):
                w_loc = kl * 2 + i
                w_glob = (k0 + kl) * n + i
                W[w_glob, col:col+r223] = W_223[w_loc]
    
    return U, V, W

def _build_2215_hybrid():
    """4x(2x2x3) + 1x(2x2x2) + 1x(2x2x1) = 40 + 7 + 3 = 50."""
    n, m, p = 2, 2, 15
    U_223, V_223, W_223 = _build_223_rank10()
    U_s, V_s, W_s = _strassen_222()
    U_r, V_r, W_r = _rank3_221()
    
    r223 = U_223.shape[1]
    r222 = U_s.shape[1]
    r221 = U_r.shape[1]
    
    total_rank = 4 * r223 + r222 + r221
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for block in range(4):
        k0 = block * 3
        U[:, col:col+r223] = U_223
        for j in range(m):
            for kl in range(3):
                v_loc = j * 3 + kl
                v_glob = j * p + (k0 + kl)
                V[v_glob, col:col+r223] = V_223[v_loc]
        for kl in range(3):
            for i in range(n):
                w_loc = kl * 2 + i
                w_glob = (k0 + kl) * n + i
                W[w_glob, col:col+r223] = W_223[w_loc]
        col += r223
    
    k0 = 12
    U[:, col:col+r222] = U_s
    for j in range(m):
        for kl in range(2):
            v_loc = j * 2 + kl
            v_glob = j * p + (k0 + kl)
            V[v_glob, col:col+r222] = V_s[v_loc]
    for kl in range(2):
        for i in range(n):
            w_loc = kl * 2 + i
            w_glob = (k0 + kl) * n + i
            W[w_glob, col:col+r222] = W_s[w_loc]
    col += r222
    
    k0 = 14
    U[:, col:col+r221] = U_r
    for j in range(m):
        v_glob = j * p + k0
        V[v_glob, col:col+r221] = V_r[j]
    for i in range(n):
        w_glob = k0 * n + i
        W[w_glob, col:col+r221] = W_r[i]
    col += r221
    
    return U, V, W

def _build_2215_strassen_only():
    """7x(2x2x2) + 1x(2x2x1) = 49 + 3 = 52."""
    n, m, p = 2, 2, 15
    U_s, V_s, W_s = _strassen_222()
    U_r, V_r, W_r = _rank3_221()
    
    r222 = U_s.shape[1]
    r221 = U_r.shape[1]
    total_rank = 7 * r222 + r221
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for pair in range(7):
        k0 = pair * 2
        U[:, col:col+r222] = U_s
        for j in range(m):
            for kl in range(2):
                v_loc = j * 2 + kl
                v_glob = j * p + (k0 + kl)
                V[v_glob, col:col+r222] = V_s[v_loc]
        for kl in range(2):
            for i in range(n):
                w_loc = kl * 2 + i
                w_glob = (k0 + kl) * n + i
                W[w_glob, col:col+r222] = W_s[w_loc]
        col += r222
    
    k0 = 14
    U[:, col:col+r221] = U_r
    for j in range(m):
        v_glob = j * p + k0
        V[v_glob, col:col+r221] = V_r[j]
    for i in range(n):
        w_glob = k0 * n + i
        W[w_glob, col:col+r221] = W_r[i]
    col += r221
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    if _remaining() < 5:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    candidates = []
    
    # Option: 5x(2x2x3) = rank 50
    if _remaining() > 10:
        try:
            U, V, W = _build_2215_blockwise()
            if _verify(U, V, W, n, m, p):
                candidates.append((U.shape[1], U, V, W))
        except Exception:
            pass
    
    # Option: 4x(2x2x3) + 1x(2x2x2) + 1x(2x2x1) = rank 50
    if _remaining() > 10:
        try:
            U, V, W = _build_2215_hybrid()
            if _verify(U, V, W, n, m, p):
                candidates.append((U.shape[1], U, V, W))
        except Exception:
            pass
    
    # Option: 7x(2x2x2) + 1x(2x2x1) = rank 52
    if _remaining() > 5:
        try:
            U, V, W = _build_2215_strassen_only()
            if _verify(U, V, W, n, m, p):
                candidates.append((U.shape[1], U, V, W))
        except Exception:
            pass
    
    # Schoolbook fallback
    if _remaining() > 2:
        U, V, W = _schoolbook(n, m, p)
        candidates.append((U.shape[1], U, V, W))
    
    if not candidates:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    candidates.sort(key=lambda x: x[0])
    _, U, V, W = candidates[0]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
