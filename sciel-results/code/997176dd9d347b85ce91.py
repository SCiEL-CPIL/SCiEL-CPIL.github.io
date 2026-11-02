import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    nrm, mpr, pnr = n*m, m*p, p*n
    
    # Target tensor T[i, j, k] = 1 if indices match multiplication, else 0
    # We only care about the 105 non-zero entries for the check
    # T_target is a list of (idx_u, idx_v, idx_w, value)
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                targets.append((u_idx, v_idx, w_idx, 1.0))
                
    # Precompute matrices for ALS
    # We want to minimize sum_t (sum_c U[u_t, c] V[v_t, c] W[w_t, c] - 1)^2
    
    deadline = time.time() + 110
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    U_sch = np.zeros((nrm, best_rank))
    V_sch = np.zeros((mpr, best_rank))
    W_sch = np.zeros((pnr, best_rank))
    for idx, (u, v, w, val) in enumerate(targets):
        U_sch[u, idx] = 1.0
        V_sch[v, idx] = 1.0
        W_sch[w, idx] = 1.0
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    
    # Try lower ranks
    for rank in range(80, 40, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(nrm, rank)
            V = np.random.randn(mpr, rank)
            W = np.random.randn(pnr, rank)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Solve U
                # For each target t: sum_c U[u_t, c] (V[v_t, c] W[w_t, c]) = 1
                # Let M_u[u, c] = V[v_t, c] W[w_t, c] for the t that has u_t=u
                # This is sparse.
                
                # Efficient way:
                # Construct A_u: (nrm * rank) x (nrm * rank)? No.
                # For each u, we have a set of equations.
                
                # Simpler: Just use the full system if small enough?
                # 105 equations, 15*rank unknowns.
                
                # Let's build the linear system for U
                # X = U.flatten()
                # A @ X = b
                # A has shape (105, nrm * rank)
                # This is small.
                
                A_U = np.zeros((len(targets), nrm * rank))
                b = np.ones(len(targets))
                
                for t, (u, v, w, val) in enumerate(targets):
                    for c in range(rank):
                        A_U[t, u * rank + c] = V[v, c] * W[w, c]
                        
                # Solve A_U @ U_flat = b
                U_flat, residuals, rank_u, s = np.linalg.lstsq(A_U, b, rcond=None)
                U = U_flat.reshape(nrm, rank)
                
                # Solve V
                A_V = np.zeros((len(targets), mpr * rank))
                for t, (u, v, w, val) in enumerate(targets):
                    for c in range(rank):
                        A_V[t, v * rank + c] = U[u, c] * W[w, c]
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = V_flat.reshape(mpr, rank)
                
                # Solve W
                A_W = np.zeros((len(targets), pnr * rank))
                for t, (u, v, w, val) in enumerate(targets):
                    for c in range(rank):
                        A_W[t, w * rank + c] = U[u, c] * V[v, c]
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = W_flat.reshape(pnr, rank)
                
            # Check error
            error = 0
            for u, v, w, val in targets:
                s = 0
                for c in range(rank):
                    s += U[u, c] * V[v, c] * W[w, c]
                error += (s - val) ** 2
                
            if error < 1e-6:
                # Snap to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Re-check after snapping
                error_snap = 0
                for u, v, w, val in targets:
                    s = 0
                    for c in range(rank):
                        s += U[u, c] * V[v, c] * W[w, c]
                    error_snap += (s - val) ** 2
                    
                if error_snap < 1e-9:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
                    
        if found:
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
