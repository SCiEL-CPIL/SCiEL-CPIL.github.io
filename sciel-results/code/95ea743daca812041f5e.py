import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 115

def _create_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    T_rec = _reconstruct(U, V, W)
    T_target = _create_target_tensor(n, m, p)
    return np.allclose(T_rec, T_target, atol=tol)

def _build_schoolbook_sub(n, m, p):
    """Build schoolbook rank decomposition for M_{n,m,p} with rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, r

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy: Decompose M_{3,4,8} into two M_{3,4,4} tensors
    # Split k index into k = k1 + 4*k2, where k1 in [0,4), k2 in [0,2)
    # Each half is a 3x4x4 multiplication tensor
    # Total rank = 2 * rank(M_{3,4,4}) = 2 * 48 = 96
    
    sub_n, sub_m, sub_p = 3, 4, 4
    U_sub, V_sub, W_sub, r_sub = _build_schoolbook_sub(sub_n, sub_m, sub_p)
    
    total_rank = 2 * r_sub
    
    U_new = np.zeros((n * m, total_rank))
    V_new = np.zeros((m * p, total_rank))
    W_new = np.zeros((p * n, total_rank))
    
    # First half: k = 0..3 (k2=0)
    # U indices: i*4 + j (same as sub)
    # V indices: j*8 + k1 for k1=0..3 -> rows 0..31 in V_new
    # W indices: k1*3 + i for k1=0..3 -> rows 0..11 in W_new
    U_new[:, 0:r_sub] = U_sub
    V_new[0:sub_m * sub_p, 0:r_sub] = V_sub
    W_new[0:sub_p * sub_n, 0:r_sub] = W_sub
    
    # Second half: k = 4..7 (k2=1)
    # U indices: i*4 + j (same as sub)
    # V indices: j*8 + (k1+4) for k1=0..3 -> rows 16..31 in V_new (j*8+4+j2 where j2=0..3)
    #   Actually V index = j*p + k = j*8 + (k1+4)
    #   For j=0: k=4,5,6,7 -> indices 4,5,6,7
    #   For j=1: k=4,5,6,7 -> indices 12,13,14,15
    #   For j=2: k=4,5,6,7 -> indices 20,21,22,23
    #   For j=3: k=4,5,6,7 -> indices 28,29,30,31
    # W indices: (k1+4)*3 + i for k1=0..3 -> rows 12..23 in W_new
    #   For k1=0: k=4, i=0,1,2 -> indices 12,13,14
    #   For k1=1: k=5, i=0,1,2 -> indices 15,16,17
    #   For k1=2: k=6, i=0,1,2 -> indices 18,19,20
    #   For k1=3: k=7, i=0,1,2 -> indices 21,22,23
    
    U_new[:, r_sub:2*r_sub] = U_sub
    
    # Map V_sub indices (j*4 + k1) to V_new indices (j*8 + k1 + 4)
    for j in range(sub_m):
        for k1 in range(sub_p):
            v_sub_idx = j * sub_p + k1
            v_new_idx = j * p + (k1 + sub_p)
            V_new[v_new_idx, r_sub:2*r_sub] = V_sub[v_sub_idx, :]
    
    # Map W_sub indices (k1*3 + i) to W_new indices ((k1+4)*3 + i)
    for k1 in range(sub_p):
        for i in range(sub_n):
            w_sub_idx = k1 * sub_n + i
            w_new_idx = (k1 + sub_p) * sub_n + i
            W_new[w_new_idx, r_sub:2*r_sub] = W_sub[w_sub_idx, :]
    
    # Verify
    if _verify_exact(U_new, V_new, W_new, n, m, p):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Fallback: full schoolbook
    U_sb, V_sb, W_sb, _ = _build_schoolbook_sub(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
