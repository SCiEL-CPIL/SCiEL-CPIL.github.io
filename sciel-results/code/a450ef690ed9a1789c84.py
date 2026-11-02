import numpy as np

def construct():
    n, m, p = 3, 5, 6
    nm, mp, pn = n*m, m*p, p*n  # 15, 30, 18
    
    # Start with schoolbook rank as upper bound
    rank = nm * mp * pn  # 90
    
    # Initialize with schoolbook
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Now try to reduce rank via iterative refinement
    # But first, let's try a smaller rank with random init and iterate
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    for trial_rank in range(20, rank + 1):
        # Try random initialization
        for attempt in range(5):
            U_try = np.random.randn(nm, trial_rank)
            V_try = np.random.randn(mp, trial_rank)
            W_try = np.random.randn(pn, trial_rank)
            
            # Iterate: fix two, solve for third
            converged = False
            for it in range(100):
                # Fix V, W; solve for U
                U_new = np.zeros_like(U_try)
                for i in range(n):
                    for j in range(m):
                        # Build p x trial_rank matrix
                        M = np.zeros((p, trial_rank))
                        for k in range(p):
                            for r in range(trial_rank):
                                M[k, r] = V_try[j*p + k, r] * W_try[k*n + i, r]
                        # Solve M @ u = ones(p)
                        try:
                            u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_try[i*m + j, :] = u
                        except:
                            U_try[i*m + j, :] = 0.0
                
                # Fix U, W; solve for V
                V_new = np.zeros_like(V_try)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, trial_rank))
                        for i in range(n):
                            for r in range(trial_rank):
                                M[i, r] = U_try[i*m + j, r] * W_try[k*n + i, r]
                        try:
                            v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_try[j*p + k, :] = v
                        except:
                            V_try[j*p + k, :] = 0.0
                
                # Fix U, V; solve for W
                W_new = np.zeros_like(W_try)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, trial_rank))
                        for j in range(m):
                            for r in range(trank if False else trial_rank):
                                M[j, r] = U_try[i*m + j, r] * V_try[j*p + k, r]
                        try:
                            w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_try[k*n + i, :] = w
                        except:
                            W_try[k*n + i, :] = 0.0
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(trial_rank):
                                val += U_try[i*m + j, r] * V_try[j*p + k, r] * W_try[k*n + i, r]
                            err += (val - 1.0) ** 2
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, trial_rank
                break
        
        if best_rank == trial_rank:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
