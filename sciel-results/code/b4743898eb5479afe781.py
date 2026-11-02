import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i*m+j, j*p+k, k*n+i))
    
    schoolbook_rank = len(entries)
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    for idx, (a, b, c) in enumerate(entries):
        U_sb[a, idx] = 1
        V_sb[b, idx] = 1
        W_sb[c, idx] = 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    
    for r in range(2, min(schoolbook_rank, 54)):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            
            num_eq = len(entries)
            num_unk = p * n * r
            
            A_mat = np.zeros((num_eq, num_unk))
            b_vec = np.ones(num_eq)
            
            for eq_idx, (a, b, c) in enumerate(entries):
                for k in range(r):
                    col = c * r + k
                    A_mat[eq_idx, col] = U[a, k] * V[b, k]
            
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            W = W_flat.reshape(p*n, r)
            
            # Verify
            valid = True
            for a, b, c in entries:
                val = np.dot(U[a,:], V[b,:] * W[c,:])
                if abs(val - 1) > 1e-6:
                    valid = False
                    break
            
            if valid:
                U_rnd = np.round(U * 2) / 2
                V_rnd = np.round(V * 2) / 2
                W_rnd = np.round(W * 2) / 2
                
                valid_rnd = True
                for a, b, c in entries:
                    val = np.dot(U_rnd[a,:], V_rnd[b,:] * W_rnd[c,:])
                    if abs(val - 1) > 1e-9:
                        valid_rnd = False
                        break
                
                if valid_rnd:
                    best_rank = r
                    best_U, best_V, best_W = U_rnd, V_rnd, W_rnd
                    found = True
                    break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
