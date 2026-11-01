import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def compute_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        return max_err
    
    # Strategy: fix U,V, solve for W; then optimize U,V
    for target_rank in range(24, rank_sb):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 31 + target_rank)
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            
            # Try to find W that satisfies all equations
            # For each (i,k), solve: A_{(i,0,k)} · w = 1, A_{(i,1,k)} · w = 1
            W = np.zeros((p*n, target_rank))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build 2×r system
                    A0 = U[i*m+0, :] * V[0*p+k, :]  # length r
                    A1 = U[i*m+1, :] * V[1*p+k, :]  # length r
                    
                    # Solve [A0; A1] · w = [1; 1]
                    A_mat = np.vstack([A0, A1])
                    b_vec = np.array([1.0, 1.0])
                    
                    # Least squares solution
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        # Check if solution is exact
                        if np.max(np.abs(A_mat @ w - b_vec)) < 1e-6:
                            W[k*n+i, :] = w
                        else:
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                
                if not feasible:
                    break
            
            if feasible:
                # Verify
                err = compute_error(U, V, W)
                if err < 1e-5:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
