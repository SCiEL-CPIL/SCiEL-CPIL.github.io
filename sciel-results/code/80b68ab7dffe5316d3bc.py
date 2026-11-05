import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 3, 3, 6
    
    # Build the 3x3x6 tensor T336
    T336 = np.zeros((9, 18, 18))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T336[a, b, c] = 1.0
    
    def _verify_full(U, V, W, tol=1e-4):
        """Verify full 3x3x6 decomposition using matrix operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n*m: return False
        if V.shape[0] != m*p: return False
        if W.shape[0] != p*n: return False
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]: return False
        
        # Build full tensor approximation
        approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    if abs(approx[a, b, c] - 1.0) > tol:
                        return False
        
        # Check that non-required entries are zero
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        if abs(approx[a, b, c]) > tol:
                            return False
        return True

    def hooi_336(rank, max_iter=50, tol=1e-6, seed=0):
        """Higher Order Orthogonal Iteration for 3x3x6 tensor."""
        np.random.seed(seed)
        # Initialize with random matrices
        U = np.random.randn(9, rank)
        V = np.random.randn(18, rank)
        W = np.random.randn(18, rank)
        
        # Normalize columns
        for i in range(rank):
            U[:, i] /= np.linalg.norm(U[:, i])
            V[:, i] /= np.linalg.norm(V[:, i])
            W[:, i] /= np.linalg.norm(W[:, i])
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # M_U = T336 * (V^T W)
            # V^T W is rank x rank
            VW = V.T @ W # (rank, rank)
            # Reshape T336 to (9, 18*18)
            T_flat = T336.reshape(9, -1)
            # M_U = T_flat @ (VW)
            M_U = T_flat @ VW
            # SVD
            U_new, s, Vt = np.linalg.svd(M_U, full_matrices=False)
            # Take first 'rank' columns
            U = U_new[:, :rank]
            
            # Update V
            # M_V = T336 * (W^T U)
            WU = W.T @ U
            T_flat_v = T336.reshape(18, -1)
            M_V = T_flat_v @ WU
            V_new, s, Vt = np.linalg.svd(M_V, full_matrices=False)
            V = V_new[:, :rank]
            
            # Update W
            # M_W = T336 * (U^T V)
            UV = U.T @ V
            T_flat_w = T336.reshape(18, -1)
            M_W = T_flat_w @ UV
            W_new, s, Vt = np.linalg.svd(M_W, full_matrices=False)
            W = W_new[:, :rank]
            
            # Check convergence
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T336 - approx)**2)
            if err < tol:
                break
        
        if _verify_full(U, V, W, tol=1e-4):
            return U, V, W
        return None

    best_result = None
    best_rank = 100
    
    # Try ranks from 28 upwards
    for rank in range(28, 40):
        if time.time() > deadline - 20:
            break
        
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            result = hooi_336(rank, max_iter=50, seed=seed * 10000 + rank)
            if result is not None:
                best_result = result
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Schoolbook rank 54
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
