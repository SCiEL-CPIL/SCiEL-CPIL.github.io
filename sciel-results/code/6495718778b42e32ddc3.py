import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 140 # 10s margin
    
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
        
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
        
    # Search for lower ranks
    # We search from high to low to find a valid solution quickly, 
    # but we keep searching lower to minimize rank.
    # Actually, searching from low to high is safer to ensure we don't miss a lower rank 
    # if we run out of time while checking a high rank that is feasible but not minimal.
    # However, high ranks are more likely to be feasible.
    # Let's search from 53 down to 30.
    
    for r in range(53, 29, -1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        # Use small integer entries for V and W
        trials = 1000 # Increase trials
        for trial in range(trials):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(12345 + r * 10000 + trial)
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            if np.linalg.matrix_rank(M) < r:
                continue
                
            try:
                T_flat = T.reshape(nm, mp * pn)
                U_transpose, residuals, rank, sv = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_transpose.T
            except:
                continue
                
            if is_half_integer(U, tol=1e-5):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V, W
                    # Do not break, continue to see if we can find even lower rank
                    # But we are iterating r downwards, so if we find at r=53, 
                    # we will check r=52 next.
                    break # Break out of trial loop for this r, move to r-1
                    
    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
