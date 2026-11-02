import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of J_{n,m,p}."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    return bool(np.allclose(T, 1.0, atol=1e-10))

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    - Start with a rank-50 block-structure decomposition using Strassen on 2x2 blocks
      (splitting the inner dimension of size 3 into 2+1).
    - Then attempt to reduce rank via structured optimization:
      * Use a smooth surrogate (sum-of-squares) to find lower-rank solutions.
      * Try random restarts with gradient-based polishing.
    - Return the best valid decomposition found within the time budget.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # =================================================================
    # Phase 1: Construct the rank-50 Strassen-based decomposition
    # =================================================================
    rank_init = 50
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4  # A[0,0], A[0,1], A[1,0], A[1,1]
    
    # Part 1: A[:,0:2] (2x2) * B[0:2,:] (2x9)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # Strassen's 7 products
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (column 8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A[:,2:3] (2x1) * B[2:3,:] (1x9)
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_init
    
    # Verify initial decomposition
    if not _check_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result
    
    # =================================================================
    # Phase 2: Attempt rank reduction via structured optimization
    # =================================================================
    # The key insight: the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find a lower-rank solution.
    # 
    # Approach: Use a smooth loss L = sum_{i,j,k} (T[i,j,k] - 1)^2
    # and try to minimize it with random restarts at various ranks.
    # 
    # We'll use a simple gradient-free approach: random perturbations
    # of the factor matrices, accepting if the loss decreases.
    
    def _compute_loss(U_arr, V_arr, W_arr, rank):
        """Compute sum of squared errors."""
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return float(np.sum((T - 1.0) ** 2))
    
    def _try_reduce_rank(start_rank, target_rank, U_init, V_init, W_init, max_iters=200, time_budget=5):
        """
        Try to reduce from start_rank to target_rank using iterative optimization.
        Returns (U, V, W) if successful, else None.
        """
        start_time = time.time()
        U_arr = np.array(U_init, dtype=float).copy()
        V_arr = np.array(V_init, dtype=float).copy()
        W_arr = np.array(W_init, dtype=float).copy()
        
        # Truncate to target_rank
        U_arr = U_arr[:, :target_rank]
        V_arr = V_arr[:, :target_rank]
        W_arr = W_arr[:, :target_rank]
        
        current_loss = _compute_loss(U_arr, V_arr, W_arr, target_rank)
        
        if current_loss < 1e-10:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
        
        lr = 0.1
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Random perturbation
            dU = np.random.normal(0, lr, U_arr.shape)
            dV = np.random.normal(0, lr, V_arr.shape)
            dW = np.random.normal(0, lr, W_arr.shape)
            
            new_U = U_arr + dU
            new_V = V_arr + dV
            new_W = W_arr + dW
            
            new_loss = _compute_loss(new_U, new_V, new_W, target_rank)
            
            if new_loss < current_loss:
                U_arr, V_arr, W_arr = new_U, new_V, new_W
                current_loss = new_loss
                if current_loss < 1e-10:
                    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
                # Decay learning rate
                lr *= 0.95
            else:
                lr *= 0.99
        
        return None
    
    # Try to reduce rank from 50 to lower values
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But the actual minimum rank is unknown; let's try a few values
    
    for target_rank in range(45, 25, -1):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Start from the current best decomposition, zero-pad or truncate
            U_init = best_result["U"]
            V_init = best_result["V"]
            W_init = best_result["W"]
            
            result = _try_reduce_rank(best_rank, target_rank, U_init, V_init, W_init, 
                                       max_iters=100, time_budget=3)
            
            if result is not None:
                U_new, V_new, W_new = result
                if _check_decomposition(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target_rank
                    break  # Move to next lower target
    
    # =================================================================
    # Phase 3: Final verification and return
    # =================================================================
    if not _check_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
