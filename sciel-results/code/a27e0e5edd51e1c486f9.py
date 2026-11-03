"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute indices for the tensor
    ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    U_idx = ii * m + jj
    V_idx = jj * p + kk
    W_idx = kk * n + ii
    
    def verify(U, V, W):
        U_sel = U[U_idx]
        V_sel = V[V_idx]
        W_sel = W[W_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)
    
    # Strassen's algorithm for 2x2x2 matrix multiplication (rank 7)
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix mult with rank 7 (Strassen)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a11+a22)(b11+b22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2: (a21+a22)b11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = 1
        
        # M3: a11(b12-b22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # M4: a22(b21-b11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # M5: (a11+a12)b22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # M6: (a21-a11)(b11+b12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7: (a12-a22)(b21+b22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    # Schoolbook for arbitrary ns x ms x ps (rank ns*ms*ps)
    def schoolbook(ns, ms, ps):
        """Return U, V, W for ns x ms x ps matrix mult with schoolbook rank."""
        rank = ns * ms * ps
        U = np.zeros((ns*ms, rank))
        V = np.zeros((ms*ps, rank))
        W = np.zeros((ps*ns, rank))
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    col = i*ms*ps + j*ps + k
                    U[i*ms+j, col] = 1
                    V[j*ps+k, col] = 1
                    W[k*ns+i, col] = 1
        return U, V, W
    
    # Build full U, V, W for a given block decomposition
    def build_decomposition(n_blocks, m_blocks, p_blocks):
        total_rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        for ni, ns in enumerate(n_blocks):
            for mi, ms in enumerate(m_blocks):
                for pi, ps in enumerate(p_blocks):
                    if ns == 2 and ms == 2 and ps == 2:
                        u, v, w = strassen_2x2x2()
                    else:
                        u, v, w = schoolbook(ns, ms, ps)
                    
                    rank = u.shape[1]
                    total_rank += rank
                    
                    n_offset = sum(n_blocks[:ni])
                    m_offset = sum(m_blocks[:mi])
                    p_offset = sum(p_blocks[:pi])
                    
                    U_list.append((n_offset, ns, ms, u))
                    V_list.append((m_offset, ms, ps, v))
                    W_list.append((p_offset, ps, ns, w))
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        col_offset = 0
        for idx in range(len(U_list)):
            n_off, ns, ms, u = U_list[idx]
            m_off, ms2, ps, v = V_list[idx]
            p_off, ps2, ns2, w = W_list[idx]
            rank = u.shape[1]
            
            for i in range(ns):
                for j in range(ms):
                    U_full[(n_off+i)*m + (m_off+j), col_offset:col_offset+rank] = u[i*ms+j, :]
            
            for j in range(ms2):
                for k in range(ps):
                    V_full[(m_off+j)*p + (p_off+k), col_offset:col_offset+rank] = v[j*ps+k, :]
            
            for k in range(ps2):
                for i in range(ns2):
                    W_full[(p_off+k)*n + (n_off+i), col_offset:col_offset+rank] = w[k*ns2+i, :]
            
            col_offset += rank
        
        return U_full, V_full, W_full, total_rank
    
    # Try different block decompositions and pick the best one
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook
    
    # Candidate decompositions
    candidates = [
        ([2, 1], [2, 2], [2, 2, 2, 1]),
        ([1, 2], [2, 2], [2, 2, 2, 1]),
        ([2, 1], [2, 2], [1, 2, 2, 2]),
        ([2, 1], [2, 2], [2, 2, 1, 2]),
        ([2, 1], [2, 2], [2, 1, 2, 2]),
        ([2, 1], [1, 3], [2, 2, 2, 1]),
        ([2, 1], [3, 1], [2, 2, 2, 1]),
        ([1, 2], [1, 3], [2, 2, 2, 1]),
        ([1, 2], [3, 1], [2, 2, 2, 1]),
        ([2, 1], [2, 1, 1], [2, 2, 2, 1]),
        ([2, 1], [1, 2, 1], [2, 2, 2, 1]),
        ([2, 1], [1, 1, 2], [2, 2, 2, 1]),
    ]
    
    for nb, mb, pb in candidates:
        if time.time() > deadline:
            break
        if sum(nb) != n or sum(mb) != m or sum(pb) != p:
            continue
        
        U_c, V_c, W_c, r = build_decomposition(nb, mb, pb)
        
        if verify(U_c, V_c, W_c) and r < best_rank:
            best_U, best_V, best_W = U_c, V_c, W_c
            best_rank = r
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
