import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_school = np.zeros((I, rank_schoolbook), dtype=int)
    V_school = np.zeros((J, rank_schoolbook), dtype=int)
    W_school = np.zeros((K, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Search for lower rank
    # Lower bound is max(I, J, K) = 24
    # Try ranks from 47 down to 30
    for rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            # Bias towards sparsity
            U_curr = np.random.randint(-1, 2, size=(I, rank))
            V_curr = np.random.randint(-1, 2, size=(J, rank))
            W_curr = np.random.randint(-1, 2, size=(K, rank))
            
            # ALS iterations with snapping
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(I):
                    # T[i, :, :] = U[i, :] @ (V ⊙ W).T
                    # Flatten T[i] to vector of length J*K
                    # Design matrix D: (J*K, rank) where D[(j,k), c] = V[j,c] * W[k,c]
                    # Note: indices in T are (i, j, k) but T is stored as (I, J, K)
                    # T[i, j, k] corresponds to T_flat[i, j*K + k]
                    
                    # Build D
                    # D has shape (J*K, rank)
                    # We can compute this efficiently
                    # V is (J, rank), W is (K, rank)
                    # Khatri-Rao product of V and W?
                    # No, we need element-wise product for each c, then flatten.
                    # D[j*K + k, c] = V[j, c] * W[k, c]
                    
                    # This is a bit slow in a loop. Let's vectorize.
                    # V_rep = V[:, np.newaxis, :] # (J, 1, rank)
                    # W_rep = W[np.newaxis, :, :] # (1, K, rank)
                    # Prod = V_rep * W_rep # (J, K, rank)
                    # D = Prod.reshape(J*K, rank)
                    
                    V_rep = V_curr[:, np.newaxis, :]
                    W_rep = W_curr[np.newaxis, :, :]
                    Prod = V_rep * W_rep
                    D = Prod.reshape(J * K, rank)
                    
                    target = T[i].flatten()
                    
                    # Solve D.T @ u = target
                    try:
                        u_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                        U_curr[i, :] = u_sol
                    except:
                        pass
                    
                # Snap U
                U_curr = np.round(U_curr)
                
                # Update V
                for j in range(J):
                    # T[:, j, :] = V[j, :] @ (U ⊙ W).T ? No.
                    # T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
                    # For fixed j, we have T[:, j, :] (shape I x K)
                    # T[i, k] = sum_c U[i,c] (V[j,c] W[k,c])
                    # Let M = T[:, j, :] (I x K)
                    # Let D be (I*K, rank) where D[i*K + k, c] = U[i,c] * W[k,c]
                    # Then M.flatten() = D.T @ v_j
                    
                    U_rep = U_curr[:, np.newaxis, :]
                    W_rep = W_curr[np.newaxis, :, :]
                    Prod = U_rep * W_rep
                    D = Prod.reshape(I * K, rank)
                    
                    target = T[:, j, :].flatten()
                    
                    try:
                        v_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                        V_curr[j, :] = v_sol
                    except:
                        pass
                        
                V_curr = np.round(V_curr)
                
                # Update W
                for k in range(K):
                    # T[:, :, k] = W[k, :] @ (U ⊙ V).T ?
                    # T[i, j, k] = sum_c (U[i,c] V[j,c]) W[k,c]
                    # For fixed k, M = T[:, :, k] (I x J)
                    # D is (I*J, rank) where D[i*J + j, c] = U[i,c] * V[j,c]
                    # M.flatten() = D.T @ w_k
                    
                    U_rep = U_curr[:, np.newaxis, :]
                    V_rep = V_curr[np.newaxis, :, :]
                    Prod = U_rep * V_rep
                    D = Prod.reshape(I * J, rank)
                    
                    target = T[:, :, k].flatten()
                    
                    try:
                        w_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                        W_curr[k, :] = w_sol
                    except:
                        pass
                        
                W_curr = np.round(W_curr)
                
                # Check exactness
                # Reconstruct T
                # T_recon[i, j, k] = sum_c U[i,c] V[j,c] W[k,c]
                # Use einsum
                T_recon = np.einsum('ic,jc,kc->ijk', U_curr, V_curr, W_curr)
                
                if np.array_equal(T_recon, T):
                    # Found exact integer solution
                    best_rank = rank
                    best_U = U_curr.tolist()
                    best_V = V_curr.tolist()
                    best_W = W_curr.tolist()
                    found = True
                    break
            
            if found:
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
