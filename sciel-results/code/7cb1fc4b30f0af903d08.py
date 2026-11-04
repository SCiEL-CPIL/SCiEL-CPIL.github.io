import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    
    start_time = time.time()
    deadline = start_time + 125.0
    
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Build the sparse tensor
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
    
    num_terms = len(tensor_entries)  # 60
    u_idx = np.array([t[0] for t in tensor_entries])
    v_idx = np.array([t[1] for t in tensor_entries])
    w_idx = np.array([t[2] for t in tensor_entries])
    
    nonzero_set = set(tensor_entries)
    
    def _validate(U, V, W, tol=1e-8):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # Check all non-zero entries
        for idx in range(num_terms):
            u, v, w = u_idx[idx], v_idx[idx], w_idx[idx]
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries (sample)
        np.random.seed(123)
        for _ in range(500):
            u = np.random.randint(0, dim_u)
            v = np.random.randint(0, dim_v)
            w = np.random.randint(0, dim_w)
            if (u, v, w) not in nonzero_set:
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val) > tol:
                    return False
        return True
    
    def _try_rank(rank, max_restarts=8, time_per_restart=15.0):
        """Try to find valid decomposition of given rank."""
        if time.time() > deadline - 20:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            np.random.seed(restart * 7919 + rank)
            
            U = np.random.randn(dim_u, rank) * 0.4
            V = np.random.randn(dim_v, rank) * 0.4
            W = np.random.randn(dim_w, rank) * 0.4
            
            best_res = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            t0 = time.time()
            
            for it in range(400):
                if time.time() - t0 > time_per_restart:
                    break
                
                A_U = V[v_idx, :] * W[w_idx, :]
                U_new = np.linalg.lstsq(A_U, np.ones(num_terms), rcond=None)[0]
                U = U_new
                
                A_V = U[u_idx, :] * W[w_idx, :]
                V_new = np.linalg.lstsq(A_V, np.ones(num_terms), rcond=None)[0]
                V = V_new
                
                A_W = U[u_idx, :] * V[v_idx, :]
                W_new = np.linalg.lstsq(A_W, np.ones(num_terms), rcond=None)[0]
                W = W_new
                
                res = 0.0
                for idx in range(num_terms):
                    u, v, w = u_idx[idx], v_idx[idx], w_idx[idx]
                    val = np.dot(U[u, :], V[v, :] * W[w, :])
                    res += (val - 1.0) ** 2
                
                if res < best_res:
                    best_res = res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_res < 1e-14:
                    break
            
            if best_res < 1e-12:
                if _validate(best_U, best_V, best_W, tol=1e-9):
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try from low ranks upward
    for rank in range(20, 60):
        if time.time() > deadline - 25:
            break
        result = _try_rank(rank)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
