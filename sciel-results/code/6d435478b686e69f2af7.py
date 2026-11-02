import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _svd_reduce(U, V, W, tol=1e-9):
    """Use SVD to detect and remove linearly dependent terms."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    removed = []
    for r in range(rank):
        if time.time() > 145:
            break
        if r in removed:
            continue
        
        other_terms = [r_prime for r_prime in range(rank) if r_prime != r]
        if len(other_terms) == 0:
            continue
            
        U_others = np.delete(U, r, axis=1)
        V_others = np.delete(V, r, axis=1)
        W_others = np.delete(W, r, axis=1)
        
        U_proj, U_residual = np.linalg.lstsq(U_others, U[:, r], rcond=None)
        U_err = np.linalg.norm(U[:, r] - U_others @ U_proj)
        
        V_proj, V_residual = np.linalg.lstsq(V_others, V[:, r], rcond=None)
        V_err = np.linalg.norm(V[:, r] - V_others @ V_proj)
        
        W_proj, W_residual = np.linalg.lstsq(W_others, W[:, r], rcond=None)
        W_err = np.linalg.norm(W[:, r] - W_others @ W_proj)
        
        if U_err < tol and V_err < tol and W_err < tol:
            removed.append(r)
    
    if not removed:
        return U, V, W, rank
    
    keep = [r for r in range(rank) if r not in removed]
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    new_rank = len(keep)
    
    return U_new, V_new, W_new, new_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100

    # Attempt Rank 34 Construction
    # 1. Strassen for cols 0,1 (7 terms)
    # 2. 3 terms for each col 2..10 (27 terms)
    rank = 34
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Strassen Block for k0=0, k1=1
    k0, k1 = 0, 1
    v_e, v_f, v_g, v_h = k0, k1, 11+k0, 11+k1
    w_c00, w_c10, w_c01, w_c11 = 2*k0, 2*k0+1, 2*k1, 2*k1+1

    # m1 = (a+d)(e+h)
    U[0, r] += 1; U[3, r] += 1
    V[v_e, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1; W[w_c11, r] += 1
    r += 1
    # m2 = (c+d)e
    U[2, r] += 1; U[3, r] += 1
    V[v_e, r] += 1
    W[w_c10, r] += 1; W[w_c11, r] += -1
    r += 1
    # m3 = a(f-h)
    U[0, r] += 1
    V[v_f, r] += 1; V[v_h, r] += -1
    W[w_c01, r] += 1; W[w_c11, r] += 1
    r += 1
    # m4 = d(g-e)
    U[3, r] += 1
    V[v_g, r] += 1; V[v_e, r] += -1
    W[w_c00, r] += 1; W[w_c10, r] += 1
    r += 1
    # m5 = (a+b)h
    U[0, r] += 1; U[1, r] += 1
    V[v_h, r] += 1
    W[w_c00, r] += -1; W[w_c01, r] += 1
    r += 1
    # m6 = (c-a)(e+f)
    U[2, r] += 1; U[0, r] += -1
    V[v_e, r] += 1; V[v_f, r] += 1
    W[w_c11, r] += 1
    r += 1
    # m7 = (b-d)(g+h)
    U[1, r] += 1; U[3, r] += -1
    V[v_g, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1
    r += 1

    # Add 3 terms for each column k=2..10
    for k in range(2, p):
        if time.time() > deadline - 5:
            break
        v_e = k
        v_f = 11 + k
        w_c0 = 2 * k
        w_c1 = 2 * k + 1
        
        # Term 1: a * e_k
        U[0, r] += 1
        V[v_e, r] += 1
        W[w_c0, r] += 1
        r += 1
        
        # Term 2: b * f_k
        U[1, r] += 1
        V[v_f, r] += 1
        W[w_c0, r] += 1
        r += 1
        
        # Term 3: (c+d) * e_k
        # This contributes to C_1k
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c1, r] += 1
        r += 1

    # Verify Rank 34
    if _verify(U, V, W):
        U_red, V_red, W_red, rank_red = _svd_reduce(U, V, W)
        if rank_red < rank and _verify(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 39 Construction
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e, v_f, v_g, v_h = k0, k1, 11+k0, 11+k1
        w_c00, w_c10, w_c01, w_c11 = 2*k0, 2*k0+1, 2*k1, 2*k1+1
        U[0, r] += 1; U[3, r] += 1; V[v_e, r] += 1; V[v_h, r] += 1; W[w_c00, r] += 1; W[w_c11, r] += 1; r += 1
        U[2, r] += 1; U[3, r] += 1; V[v_e, r] += 1; W[w_c10, r] += 1; W[w_c11, r] += -1; r += 1
        U[0, r] += 1; V[v_f, r] += 1; V[v_h, r] += -1; W[w_c01, r] += 1; W[w_c11, r] += 1; r += 1
        U[3, r] += 1; V[v_g, r] += 1; V[v_e, r] += -1; W[w_c00, r] += 1; W[w_c10, r] += 1; r += 1
        U[0, r] += 1; U[1, r] += 1; V[v_h, r] += 1; W[w_c00, r] += -1; W[w_c01, r] += 1; r += 1
        U[2, r] += 1; U[0, r] += -1; V[v_e, r] += 1; V[v_f, r] += 1; W[w_c11, r] += 1; r += 1
        U[1, r] += 1; U[3, r] += -1; V[v_g, r] += 1; V[v_h, r] += 1; W[w_c00, r] += 1; r += 1

    for block in range(5):
        add_strassen_block(block*2, block*2+1)
    
    k2 = 10
    v_e2, v_g2 = k2, 11+k2
    w_c02, w_c12 = 2*k2, 2*k2+1
    U[0, r] += 1; V[v_e2, r] += 1; W[w_c02, r] += 1; r += 1
    U[1, r] += 1; V[v_g2, r] += 1; W[w_c02, r] += 1; r += 1
    U[2, r] += 1; V[v_e2, r] += 1; W[w_c12, r] += 1; r += 1
    U[3, r] += 1; V[v_g2, r] += 1; W[w_c12, r] += 1; r += 4

    if _verify(U, V, W):
        U_red, V_red, W_red, rank_red = _svd_reduce(U, V, W)
        if rank_red < rank and _verify(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback Schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
