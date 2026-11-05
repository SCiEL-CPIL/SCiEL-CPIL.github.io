"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 115  # 115s wall-clock limit (5s safety margin)

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def _build_strassen_54(n=2, m=5, p=6):
    """Build the rank-54 Strassen-based decomposition."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        # 2x1 * 1x2 -> 2x2, requires 4 scalar multiplications
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Naive blocks for j=4, k-pairs (0,1), (2,3), (4,5)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    # Trim to actual rank used
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    return U, V, W, col

def _verify_fast(U_list, V_list, W_list, n, m, p, n_trials=3):
    """Fast verification using numpy with multiple random trials."""
    U_np = np.array(U_list, dtype=float)
    V_np = np.array(V_list, dtype=float)
    W_np = np.array(W_list, dtype=float)
    r = U_np.shape[1]
    
    for trial in range(n_trials):
        np.random.seed(42 + trial * 7)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            U_r = U_np[:, r_idx].reshape(n, m)
            V_r = V_np[:, r_idx].reshape(m, p)
            W_r = W_np[:, r_idx].reshape(p, n)
            UA = U_r * A_test
            VB = V_r * B_test
            temp = UA @ VB
            C_computed += temp * W_r.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _build_schoolbook(n=2, m=5, p=6):
    """Build the naive rank n*m*p decomposition."""
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U_school[i*m+j][idx] = 1
                V_school[j*p+k][idx] = 1
                W_school[k*n+i][idx] = 1
    return U_school, V_school, W_school, rank_school

def _try_rank_reduction(U, V, W, n, m, p, start_rank, min_rank, time_limit):
    """
    Try to reduce rank below start_rank using alternating least squares.
    Returns (U_opt, V_opt, W_opt, achieved_rank) or None.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_U, best_V, best_W, best_rank = None, None, None, start_rank
    
    for target_rank in range(start_rank - 1, min_rank - 1, -1):
        if _time_remaining() < time_limit:
            break
        
        # Multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if _time_remaining() < time_limit:
                break
            
            np.random.seed(42 + restart * 13)
            
            # Initialize with truncated current solution plus noise
            if start_rank >= target_rank:
                U_opt = np.array(U_np[:, :target_rank], dtype=float).copy()
                V_opt = np.array(V_np[:, :target_rank], dtype=float).copy()
                W_opt = np.array(W_np[:, :target_rank], dtype=float).copy()
                # Add small noise to escape local minima
                U_opt += np.random.randn(n*m, target_rank) * 0.01
                V_opt += np.random.randn(m*p, target_rank) * 0.01
                W_opt += np.random.randn(p*n, target_rank) * 0.01
            else:
                U_opt = np.random.randn(n*m, target_rank) * 0.1
                V_opt = np.random.randn(m*p, target_rank) * 0.1
                W_opt = np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating least squares
            converged = False
            for iteration in range(100):
                if _time_remaining() < time_limit:
                    break
                
                # Update U: for each (i,j), solve M @ x = ones(p)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V_opt[j*p+k, :] * W_opt[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_opt[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve M @ x = ones(n)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U_opt[i*m+j, :] * W_opt[k*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_opt[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve M @ x = ones(m)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U_opt[i*m+j, :] * V_opt[j*p+k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_opt[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence and validity
                if iteration % 10 == 0:
                    if _verify_fast(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, n_trials=2):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U_opt.tolist()
                            best_V = V_opt.tolist()
                            best_W = W_opt.tolist()
                        converged = True
                        break
            
            if converged:
                break
    
    if best_rank < start_rank:
        return best_U, best_V, best_W, best_rank
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank-54 decomposition
    U, V, W, rank = _build_strassen_54(n, m, p)
    
    # Quick verification
    if not _verify_fast(U, V, W, n, m, p, n_trials=3):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, _ = _build_schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to reduce rank
    if _time_remaining() > 20:
        result = _try_rank_reduction(U, V, W, n, m, p, rank, 40, time_limit=15)
        if result is not None:
            U, V, W, rank = result
    
    # Final verification
    if not _verify_fast(U, V, W, n, m, p, n_trials=3):
        U_sb, V_sb, W_sb, _ = _build_schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
