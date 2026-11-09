import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomp(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    # Reshape to 3D arrays
    U_3d = np.array(U).reshape(n, m, rank)
    V_3d = np.array(V).reshape(m, p, rank)
    W_3d = np.array(W).reshape(p, n, rank)
    
    # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want this to be 1 for all i,j,k
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # dot product over r
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                error += (val - 1.0)**2
    return error

def _greedy_search(n, m, p, target_rank, num_trials=10, timeout=50):
    start = time.time()
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
            
        # Initialize U, V, W as list of vectors
        U = []
        V = []
        W = []
        
        # Residual is the full tensor T
        # T[i,j,k] = 1
        residual = np.ones((n, m, p))
        
        for r in range(target_rank):
            # Power iteration to find dominant mode
            # Init
            u = np.random.randn(n, m)
            v = np.random.randn(m, p)
            w = np.random.randn(p, n)
            
            for _ in range(20):
                # Update u
                # u[i,j] = sum_k residual[i,j,k] * v[j,k] * w[k,i]
                # This is a contraction
                # Let's do it efficiently
                temp_v_w = np.einsum('jk,ki->ji', v, w) # shape (m, n)
                # u[i,j] = sum_k residual[i,j,k] * temp_v_w[j,k]? No.
                # v[j,k] * w[k,i] is sum_k v[j,k] w[k,i]
                # So temp is (m, n).
                # u[i,j] = sum_k residual[i,j,k] * (v*w)[j,k] is wrong index.
                # v is (m,p), w is (p,n).
                # v[j,k] w[k,i] -> sum_k v[j,k] w[k,i] is (m,n).
                # Let M_vw = v @ w. Shape (m, n).
                # u[i,j] = sum_k residual[i,j,k] * M_vw[j,k]? No, k is in v and w.
                # v[j,k] and w[k,i].
                # So for fixed i,j: sum_k residual[i,j,k] v[j,k] w[k,i].
                # This is not a simple matrix multiply.
                
                # Let's use explicit loops for clarity, it's small.
                u_new = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        # sum_k residual[i,j,k] * v[j,k] * w[k,i]
                        u_new[i, j] = np.dot(residual[i, j, :], v[j, :] * w[:, i])
                u = u_new / (np.linalg.norm(u_new) + 1e-9)
                
                v_new = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        # sum_i residual[i,j,k] * u[i,j] * w[k,i]
                        v_new[j, k] = np.dot(residual[:, j, k], u[:, j] * w[k, :])
                v = v_new / (np.linalg.norm(v_new) + 1e-9)
                
                w_new = np.zeros((p, n))
                for k in range(p):
                    for i in range(n):
                        # sum_j residual[i,j,k] * u[i,j] * v[j,k]
                        w_new[k, i] = np.dot(residual[i, :, k], u[i, :] * v[:, k])
                w = w_new / (np.linalg.norm(w_new) + 1e-9)
            
            # Add to decomposition
            U.append(u.flatten())
            V.append(v.flatten())
            W.append(w.flatten())
            
            # Update residual
            # residual -= u \otimes v \otimes w
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= u[i, j] * v[j, k] * w[k, i]
            
            # Check convergence
            if np.linalg.norm(residual) < 1e-5:
                break
        
        # Check if we found a decomposition
        U_arr = np.array(U).T # (rank, n*m)
        V_arr = np.array(V).T # (rank, m*p)
        W_arr = np.array(W).T # (rank, p*n)
        
        # Round to half-integers
        U_r = np.round(U_arr * 2) / 2.0
        V_r = np.round(V_arr * 2) / 2.0
        W_r = np.round(W_arr * 2) / 2.0
        
        error = _verify_decomp(U_r, V_r, W_r, n, m, p)
        if error < 1e-8:
            return {
                'U': U_r.tolist(),
                'V': V_r.tolist(),
                'W': W_r.tolist()
            }
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try ranks
    for target_rank in range(48, 96):
        if time.time() > deadline:
            break
        result = _greedy_search(n, m, p, target_rank, num_trials=5, timeout=10)
        if result is not None:
            return result
            
    # Fallback
    return _schoolbook_decomp(n, m, p)

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
