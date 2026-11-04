import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = 4, 4, 8
    
    # Build rank-98 Strassen decomposition
    U98, V98, W98 = _build_rank98()
    
    # Verify the rank-98 decomposition
    if not _verify(U98, V98, W98, n, m, p):
        return _naive(n, m, p)
    
    best_rank = 98
    best_U, best_V, best_W = U98, V98, W98
    
    if time.time() < deadline - 10.0:
        reduced = _try_reduce_rank(U98, V98, W98, n, m, p, deadline)
        if reduced is not None:
            best_rank, best_U, best_V, best_W = reduced
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank98():
    n, m, p = 4, 4, 8
    
    # 2x2x2 Strassen matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 Strassen via Kronecker product (rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: columns 0:49
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Block 2: columns 49:98
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _naive(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to find a lower-rank equivalent decomposition.
    
    The key structural observation: U[:, 0:49] == U[:, 49:98] = U4.
    This means both output column blocks use the same left factor.
    
    We try to exploit this by checking if the 98 basis tensors have
    linear dependencies that allow rank reduction.
    """
    start = time.time()
    dim_A = n * m  # 16
    dim_B = m * p  # 32
    dim_C = p * n  # 32
    rank0 = U.shape[1]  # 98
    
    # Build the full tensor T
    T = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for r in range(rank0):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Compute lower bounds from unfoldings
    T1 = T.reshape(dim_A, dim_B * dim_C)
    T2 = T.transpose(1,0,2).reshape(dim_B, dim_A * dim_C)
    T3 = T.transpose(2,0,1).reshape(dim_C, dim_A * dim_B)
    
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    r3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    lower_bound = max(r1, r2, r3)
    
    if time.time() > deadline - 15.0:
        return None
    
    # The tensor has structure: T = sum_{r=0}^{48} U4[:,r] ⊗ M_r
    # where M_r = V1[:,r] ⊗ W1[:,r] + V2[:,r] ⊗ W2[:,r] is a 32x32 matrix.
    # Each M_r has rank at most 2 (sum of two rank-1 matrices).
    # So T = sum_{r=0}^{48} sum_{s=1}^{2} U4[:,r] ⊗ (M_r)_s
    # where (M_r)_s are rank-1 components of M_r.
    # This gives at most 48*2 = 96 rank-1 terms if all M_r have rank 2.
    # But some M_r might have rank 1, giving even fewer terms.
    
    # Let's check the rank of each M_r
    U4 = U[:, 0:49]
    V1 = V[:, 0:49]
    W1 = W[:, 0:49]
    V2 = V[:, 49:98]
    W2 = W[:, 49:98]
    
    total_rank = 0
    new_U = []
    new_V = []
    new_W = []
    
    for r in range(49):
        if time.time() > deadline - 5.0:
            return None
        
        # M_r = V1[:,r] ⊗ W1[:,r] + V2[:,r] ⊗ W2[:,r]
        M_r = np.outer(V1[:, r], W1[:, r]) + np.outer(V2[:, r], W2[:, r])
        
        # Find rank of M_r
        rank_M = np.linalg.matrix_rank(M_r, tol=1e-10)
        
        if rank_M == 0:
            continue
        elif rank_M == 1:
            # M_r is rank-1: M_r = v ⊗ w
            # Find v and w
            # Use SVD
            U_svd, s, Vt_svd = np.linalg.svd(M_r, full_matrices=False)
            v = U_svd[:, 0] * np.sqrt(s[0])
            w = Vt_svd[0, :] * np.sqrt(s[0])
            
            new_U.append(U4[:, r])
            new_V.append(v)
            new_W.append(w)
            total_rank += 1
        else:
            # M_r is rank-2: M_r = v1 ⊗ w1 + v2 ⊗ w2
            # Use SVD to get the two rank-1 components
            U_svd, s, Vt_svd = np.linalg.svd(M_r, full_matrices=False)
            
            # Take the top 2 components
            for comp in range(min(2, len(s))):
                if s[comp] > 1e-10:
                    v = U_svd[:, comp] * np.sqrt(s[comp])
                    w = Vt_svd[comp, :] * np.sqrt(s[comp])
                    
                    new_U.append(U4[:, r])
                    new_V.append(v)
                    new_W.append(w)
                    total_rank += 1
    
    if total_rank < rank0 and total_rank > 0:
        U_new = np.column_stack(new_U)
        V_new = np.column_stack(new_V)
        W_new = np.column_stack(new_W)
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p):
            return (total_rank, U_new, V_new, W_new)
    
    return None
