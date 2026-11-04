import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank-33 solution: 12 schoolbook + 21 Strassen
    rank_start = 33
    U = np.zeros((n * m, rank_start))
    V = np.zeros((m * p, rank_start))
    W = np.zeros((p * n, rank_start))
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Three 2x2x2 Strassen blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00+A11)(B00+B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2: (A10+A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3: A00(B10-B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4: A11(B01-B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5: (A00+A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
    
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Fast vectorized verification using einsum."""
        r = U_mat.shape[1]
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)
    
    # Verify initial solution
    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii * m + jj, col_s] = 1.0
                    V_s[jj * p + kk, col_s] = 1.0
                    W_s[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank_start
    
    # Try to reduce rank by removing columns
    # First, try removing single columns from the rank-33 solution
    for trial_rank in range(rank_start - 1, max(20, rank_start - 8), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Try all single-column removals first for rank_start-1
        if trial_rank == rank_start - 1:
            for col_idx in range(rank_start):
                if time.time() > deadline - 5:
                    break
                mask = np.ones(rank_start, dtype=bool)
                mask[col_idx] = False
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                if verify_fast(U_test, V_test, W_test):
                    best_rank = trial_rank
                    best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                    found = True
                    break
        
        # Try random subsets for lower ranks
        if not found and trial_rank < rank_start - 1:
            np.random.seed(42 + trial_rank)
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                cols = np.random.choice(best_rank, trial_rank, replace=False)
                U_sub = best_U[:, cols]
                V_sub = best_V[:, cols]
                W_sub = best_W[:, cols]
                
                if verify_fast(U_sub, V_sub, W_sub):
                    best_rank = trial_rank
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                    found = True
                    break
        
        if found:
            # Continue trying to reduce further
            continue
        else:
            # Try a few more random subsets before giving up on this rank
            np.random.seed(100 + trial_rank)
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                cols = np.random.choice(best_rank, trial_rank, replace=False)
                U_sub = best_U[:, cols]
                V_sub = best_V[:, cols]
                W_sub = best_W[:, cols]
                
                if verify_fast(U_sub, V_sub, W_sub):
                    best_rank = trial_rank
                    best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                    found = True
                    break
            if not found:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
