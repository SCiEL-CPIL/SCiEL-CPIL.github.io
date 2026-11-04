import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    T = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Flatten T for ALS
    T_flat = T.reshape(size_u, size_v * size_w)
    
    best_result = None
    best_rank = n * m * p
    
    # Initialize with schoolbook
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n)
    lower_bound = max(size_u, size_v, size_w)
    
    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Random initialization
        U = np.random.randn(size_u, rank)
        V = np.random.randn(size_v, rank)
        W = np.random.randn(size_w, rank)
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Update U
            # K is (size_v * size_w) x rank
            # K[a, r] = V[b, r] * W[c, r] where a = b * size_w + c
            # We can compute K efficiently
            # V is size_v x rank, W is size_w x rank
            # We want outer product of V and W for each r?
            # K[:, r] = V[:, r] outer W[:, r] flattened
            # This is equivalent to:
            # K = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            K_U = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
            # Solve K_U.T @ U.T = T_flat
            # T_flat is size_u x (size_v * size_w)
            # We want U (size_u x rank)
            # Equation: sum_r U[i, r] * K_U[j, r] = T_flat[i, j]
            # This is: K_U.T @ U[i, :].T = T_flat[i, :].T
            # So U[i, :].T = lstsq(K_U.T, T_flat[i, :].T)
            # We can stack all i:
            # K_U.T (rank x (size_v*size_w))
            # T_flat.T ((size_v*size_w) x size_u)
            # Result U.T (rank x size_u)
            try:
                U = np.linalg.lstsq(K_U.T, T_flat.T, rcond=None)[0].T
            except:
                break
                
            # Update V
            # K_V is (size_u * size_w) x rank
            # K_V[a, r] = U[i, r] * W[k, r] where a = i * size_w + k
            K_V = np.einsum('ir,kr->ikr', U, W).reshape(-1, rank)
            # T_V is size_v x (size_u * size_w)
            # T_V[j, i*size_w+k] = T[i*m+j, j*p+k, k*n+i]
            # We need to reshape T to match this ordering.
            # T is (size_u, size_v, size_w)
            # We want (size_v, size_u, size_w) -> (size_v, size_u * size_w)
            T_V = T.transpose(1, 0, 2).reshape(size_v, -1)
            try:
                V = np.linalg.lstsq(K_V.T, T_V.T, rcond=None)[0].T
            except:
                break
                
            # Update W
            # K_W is (size_u * size_v) x rank
            # K_W[a, r] = U[i, r] * V[j, r] where a = i * size_v + j
            K_W = np.einsum('ir,jr->ijr', U, V).reshape(-1, rank)
            # T_W is size_w x (size_u * size_v)
            # T_W[k, i*size_v+j] = T[i*m+j, j*p+k, k*n+i]
            # T is (size_u, size_v, size_w)
            # We want (size_w, size_u, size_v) -> (size_w, size_u * size_v)
            T_W = T.transpose(2, 0, 1).reshape(size_w, -1)
            try:
                W = np.linalg.lstsq(K_W.T, T_W.T, rcond=None)[0].T
            except:
                break
                
            # Check error
            # Error = || sum_r U[i,r] V[j,r] W[k,r] - T[i,j,k] ||_F
            # We can compute the reconstructed tensor
            # Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            # err = np.linalg.norm(Rec - T)
            # To save time, we can just check a few random points or compute norm every few iterations
            if it % 5 == 0:
                Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(Rec - T)
                if err < 1e-5:
                    converged = True
                    break
                    
        if converged:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
            
    return best_result
