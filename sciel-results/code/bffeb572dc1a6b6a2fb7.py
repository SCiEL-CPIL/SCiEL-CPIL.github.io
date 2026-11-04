import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n, m, p) with all 1s
    # We need U, V, W such that sum_alpha U[i, alpha] V[j, alpha] W[k, alpha] = 1
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n, rank_sb))
    V_sb = np.zeros((m, rank_sb))
    W_sb = np.zeros((p, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i, col] = 1
                V_sb[j, col] = 1
                W_sb[k, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find lower rank
    # We search for integer A, B, C such that sum A B C = 8
    # U = A/2, etc.
    
    for r in range(best_rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        max_attempts = 200
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate random A (n x r) and B (m x r) with small integers
            A = np.random.randint(-1, 2, size=(n, r))
            B = np.random.randint(-1, 2, size=(m, r))
            
            if np.all(A == 0) or np.all(B == 0):
                continue
                
            C = np.zeros((p, r))
            valid_k = True
            
            for k in range(p):
                V_mat = np.zeros((n * m, r))
                y_vec = np.full(n * m, 8)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        V_mat[idx, :] = A[i, :] * B[j, :]
                        idx += 1
                        
                # Solve V_mat C_k = y_vec
                sol, residuals, rank, sv = np.linalg.lstsq(V_mat, y_vec, rcond=None)
                
                # Round to nearest integer
                C_k_int = np.round(sol).astype(int)
                
                # Verify exactness
                if not np.allclose(V_mat @ C_k_int, y_vec, atol=0):
                    valid_k = False
                    break
                else:
                    C[k, :] = C_k_int
            
            if valid_k:
                U = A / 2.0
                V = B / 2.0
                W = C / 2.0
                
                # Verify the decomposition
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for alpha in range(r):
                                val += U[i, alpha] * V[j, alpha] * W[k, alpha]
                            if abs(val - 1.0) > 1e-9:
                                success = False
                                break
                        if not success: break
                    if not success: break
                
                if success:
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we fail to find a decomposition for rank r, we might stop if r is small?
            # No, just continue.
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}
