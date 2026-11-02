import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Precompute tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_entries.append((i*m+j, j*p+k, k*n+i))
    
    num_entries = len(tensor_entries)
    target = np.ones(num_entries)
    
    # Precompute masks
    mask_U = {a: [idx for idx, (aa,bb,dd) in enumerate(tensor_entries) if aa==a] for a in range(n*m)}
    mask_V = {b: [idx for idx, (aa,bb,dd) in enumerate(tensor_entries) if bb==b] for b in range(m*p)}
    mask_W = {d: [idx for idx, (aa,bb,dd) in enumerate(tensor_entries) if dd==d] for d in range(p*n)}
    
    # Schoolbook
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for col, (a,b,d) in enumerate(tensor_entries):
        U_sb[a, col] = 1
        V_sb[b, col] = 1
        W_sb[d, col] = 1
    
    best = (rank_sb, U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    def compute_loss(U, V, W):
        T_hat = np.array([np.dot(U[a,:], V[b,:]*W[d,:]) for a,b,d in tensor_entries])
        return np.sum((T_hat - target)**2)
    
    # Try decreasing ranks
    for rank in range(20, rank_sb):
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(99 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # ALS: update U
                for a in range(n*m):
                    mask = mask_U[a]
                    if not mask: continue
                    M = np.column_stack([V[tensor_entries[idx][1],:] * W[tensor_entries[idx][2],:] for idx in mask])
                    try:
                        U[a,:] = np.linalg.lstsq(M, target[mask], rcond=None)[0]
                    except: pass
                
                # Update V
                for b in range(m*p):
                    mask = mask_V[b]
                    if not mask: continue
                    M = np.column_stack([U[tensor_entries[idx][0],:] * W[tensor_entries[idx][2],:] for idx in mask])
                    try:
                        V[b,:] = np.linalg.lstsq(M, target[mask], rcond=None)[0]
                    except: pass
                
                # Update W
                for d in range(p*n):
                    mask = mask_W[d]
                    if not mask: continue
                    M = np.column_stack([U[tensor_entries[idx][0],:] * V[tensor_entries[idx][1],:] for idx in mask])
                    try:
                        W[d,:] = np.linalg.lstsq(M, target[mask], rcond=None)[0]
                    except: pass
                
                loss = compute_loss(U, V, W)
                if loss < 1e-10:
                    best = (rank, U.copy(), V.copy(), W.copy())
                    break
        
        if best[0] == rank:
            break
    
    r, U, V, W = best
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
