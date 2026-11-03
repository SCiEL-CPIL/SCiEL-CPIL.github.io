import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor non-zero entries
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                entries.append((u, v, w))
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx, (u, v, w) in enumerate(entries):
        U_school[u, idx] = 1
        V_school[v, idx] = 1
        W_school[w, idx] = 1
        
    def to_list(mat):
        # Round to nearest 0.5
        rounded = np.round(mat * 2) / 2
        out = []
        for row in rounded:
            out_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    out_row.append(int(round(x)))
                else:
                    out_row.append(float(x))
            out.append(out_row)
        return out

    def verify(U, V, W):
        rank = U.shape[1]
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Check entries
        for u, v, w in entries:
            if not np.isclose(T_recon[u, v, w], 1.0, atol=1e-9):
                return False
        
        # Check zeros
        mask = np.zeros((U_dim, V_dim, W_dim), dtype=bool)
        for u, v, w in entries:
            mask[u, v, w] = True
            
        if not np.all(np.isclose(T_recon[~mask], 0.0, atol=1e-9)):
            return False
        return True

    # Pre-compute indices for faster ALS
    U_idx = []
    for u in range(U_dim):
        vs = []
        ws = []
        for k in range(p):
            v = (u % m) * p + k
            w = k * n + (u // m)
            vs.append(v)
            ws.append(w)
        U_idx.append((vs, ws))
        
    V_idx = []
    for v in range(V_dim):
        j = v // p
        k = v % p
        us = []
        ws = []
        for i in range(n):
            u = i * m + j
            w = k * n + i
            us.append(u)
            ws.append(w)
        V_idx.append((us, ws))
        
    W_idx = []
    for w in range(W_dim):
        k = w // n
        i = w % n
        us = []
        vs = []
        for j in range(m):
            u = i * m + j
            v = j * p + k
            us.append(u)
            vs.append(v)
        W_idx.append((us, vs))

    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Search ranks from 31 to 44
    for rank in range(31, 45):
        if time.time() > deadline:
            break
            
        found = False
        restarts = 30 if rank <= 33 else 10
        iterations = 150
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            if restart % 2 == 0:
                U = np.random.randn(U_dim, rank)
                V = np.random.randn(V_dim, rank)
                W = np.random.randn(W_dim, rank)
            else:
                U = np.random.choice([0, 1, -1], size=(U_dim, rank))
                V = np.random.choice([0, 1, -1], size=(V_dim, rank))
                W = np.random.choice([0, 1, -1], size=(W_dim, rank))
                U /= (np.linalg.norm(U) * np.sqrt(rank) + 1e-9)
                V /= (np.linalg.norm(V) * np.sqrt(rank) + 1e-9)
                W /= (np.linalg.norm(W) * np.sqrt(rank) + 1e-9)
                
            # ALS iterations
            for it in range(iterations):
                # Update U
                for u in range(U_dim):
                    vs, ws = U_idx[u]
                    A = np.zeros((p, rank))
                    for idx in range(p):
                        A[idx, :] = V[vs[idx], :] * W[ws[idx], :]
                    b = np.ones(p)
                    try:
                        U[u, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                        
                # Update V
                for v in range(V_dim):
                    us, ws = V_idx[v]
                    A = np.zeros((n, rank))
                    for idx in range(n):
                        A[idx, :] = U[us[idx], :] * W[ws[idx], :]
                    b = np.ones(n)
                    try:
                        V[v, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

                # Update W
                for w in range(W_dim):
                    us, vs = W_idx[w]
                    A = np.zeros((m, rank))
                    for idx in range(m):
                        A[idx, :] = U[us[idx], :] * V[vs[idx], :]
                    b = np.ones(m)
                    try:
                        W[w, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                        
            # Check error
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err_val = 0
            for u, v, w in entries:
                err_val = max(err_val, abs(T_recon[u, v, w] - 1))
            mask = np.zeros((U_dim, V_dim, W_dim), dtype=bool)
            for u, v, w in entries:
                mask[u, v, w] = True
            err_val = max(err_val, np.max(np.abs(T_recon[~mask])))
            
            if err_val < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_rank = rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
                    
        if found:
            break
            
    if best_rank == rank_school:
        U_out, V_out, W_out = U_school, V_school, W_school
    else:
        U_out, V_out, W_out = best_U, best_V, best_W
        
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
