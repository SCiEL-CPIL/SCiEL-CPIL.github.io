import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Precompute target tensor T
    # T[i, j, k] = 1 if valid, else 0
    # For standard matrix multiplication, all are valid.
    T = np.ones((n, m, p))
    
    # Search for minimal rank
    # Lower bound is 1, upper bound is schoolbook - 1
    # We iterate from low to high.
    for target_rank in range(1, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try to find decomposition of rank target_rank
        result = _greedy_cp_decomposition(T, target_rank, n, m, p, time_limit=5.0)
        
        if result is not None:
            U, V, W = result
            # Validate
            if _validate(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = (U, V, W)
                break
                
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return (U, V, W)

def _validate(U, V, W, n, m, p):
    # Reconstruct T
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    R = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Reshape u to (n, m), v to (m, p), w to (p, n)
        u_2d = u.reshape((n, m))
        v_2d = v.reshape((m, p))
        w_2d = w.reshape((p, n))
        
        # Compute contribution
        for i in range(n):
            for j in range(m):
                u_val = u_2d[i, j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_2d[j, k]
                    if v_val == 0: continue
                    w_val = w_2d[k, i]
                    T_recon[i, j, k] += u_val * v_val * w_val
                    
    # Check against T (which is all 1s)
    T_target = np.ones((n, m, p))
    
    if np.allclose(T_recon, T_target, atol=1e-6, rtol=1e-6):
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Re-check with rounded values
        T_recon_rounded = np.zeros((n, m, p))
        for r in range(R):
            u = U_rounded[:, r]
            v = V_rounded[:, r]
            w = W_rounded[:, r]
            u_2d = u.reshape((n, m))
            v_2d = v.reshape((m, p))
            w_2d = w.reshape((p, n))
            for i in range(n):
                for j in range(m):
                    u_val = u_2d[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_2d[j, k]
                        if v_val == 0: continue
                        w_val = w_2d[k, i]
                        T_recon_rounded[i, j, k] += u_val * v_val * w_val
                        
        if np.allclose(T_recon_rounded, T_target, atol=1e-9, rtol=1e-9):
            return True
    return False

def _greedy_cp_decomposition(T, rank, n, m, p, time_limit=5.0):
    """
    Greedy CP decomposition.
    T is (n, m, p).
    Returns (U, V, W) or None.
    """
    start_time = time.time()
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(rank):
        if time.time() > start_time + time_limit:
            return None
            
        # Power iteration to find best rank-1 approx of residual
        # Initialize randomly
        u = np.random.rand(n, m)
        v = np.random.rand(m, p)
        w = np.random.rand(p, n)
        
        # Normalize
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)
        
        converged = False
        for it in range(1000):
            # Update u
            Vw = v @ w # (m, p) @ (p, n) -> (m, n)
            
            u_new = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    vec = v[j, :] * w[:, i]
                    u_new[i, j] = np.dot(residual[i, j, :], vec)
            
            u_new /= (np.linalg.norm(u_new) + 1e-9)
            
            # Update v
            v_new = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    vec = u[:, j] * w[k, :]
                    v_new[j, k] = np.dot(residual[:, j, k], vec)
            
            v_new /= (np.linalg.norm(v_new) + 1e-9)
            
            # Update w
            w_new = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    vec = u[i, :] * v[:, k]
                    w_new[k, i] = np.dot(residual[i, :, k], vec)
            
            w_new /= (np.linalg.norm(w_new) + 1e-9)
            
            # Check convergence
            if np.allclose(u, u_new, atol=1e-8) and np.allclose(v, v_new, atol=1e-8) and np.allclose(w, w_new, atol=1e-8):
                u, v, w = u_new, v_new, w_new
                converged = True
                break
            
            u, v, w = u_new, v_new, w_new
        
        # Subtract this rank-1 term from residual
        term = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_val = u[i, j]
                if u_val == 0: continue
                for k in range(p):
                    term[i, j, k] = u_val * v[j, k] * w[k, i]
                    
        residual -= term
        
        U_list.append(u.flatten())
        V_list.append(v.flatten())
        W_list.append(w.flatten())
        
        # Check if residual is zero
        if np.allclose(residual, 0, atol=1e-8):
            break
            
    if len(U_list) < rank:
        pass
        
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return (U, V, W)
