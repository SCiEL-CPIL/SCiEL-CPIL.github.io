import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target rank for 2x3x3 block is 7. Total rank 3*7=21.
    target_rank_block = 7
    total_rank = 3 * target_rank_block
    
    # Try to find rank-7 decomposition for 2x3x3
    # Tensor T233[a, b, c] where a=i*m+j, b=j'*3+k, c=k'*n+i'
    # T233[i*m+j, j'*3+k, k'*n+i'] = 1 if i=i', j=j', k=k' else 0
    
    def find_233_decomposition():
        n2, m2, p2 = 2, 3, 3
        nm2 = n2 * m2
        mp2 = m2 * p2
        np2 = n2 * p2
        r = target_rank_block
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(nm2, r)
        V = np.random.randn(mp2, r)
        W = np.random.randn(np2, r)
        
        # Alternating Least Squares / Gradient Descent
        # Minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        # This is equivalent to minimizing || T_unrolled - (U*V) * W || or similar.
        # We can use a simple iterative method.
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(5000):
            if time.time() > deadline:
                break
                
            # Update U
            # For each a, U[a, :] = argmin_U sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # This is a linear least squares problem for each a.
            # Let M_U[a, b, c] = T[a, b, c]
            # We want U[a, r] such that sum_r U[a, r] (V[:, r] * W[:, r])^T approximates T[a, :, :]
            # Let B_c = V[:, r] * W[:, r] (vector of length mp2*np2? No, V is mp2 x r, W is np2 x r)
            # The term is sum_r U[a, r] * V[b, r] * W[c, r]
            # We can flatten V and W? No.
            # Construct matrix M_U of size (mp2 * np2) x r
            # M_U[b * np2 + c, r] = V[b, r] * W[c, r]
            M_U = np.einsum('br,cr->bcr', V, W).reshape(mp2 * np2, r)
            # Reshape T for this a
            T_a = np.zeros((mp2, np2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        if i == i and j == j and k == k: # This is always true, just filling T
                            pass
            # Actually, T233 is sparse.
            # T233[i*m2+j, j'*p2+k, k'*n2+i'] = 1 iff i=i', j=j', k=k'
            # Let's build T233 explicitly
            T233 = np.zeros((nm2, mp2, np2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        a = i * m2 + j
                        b = j * p2 + k
                        c = k * n2 + i
                        T233[a, b, c] = 1.0
            
            # Now update U
            for a in range(nm2):
                # Target vector t_a of length mp2 * np2
                t_a = T233[a, :, :].flatten()
                # Solve M_U @ u = t_a
                try:
                    U[a, :] = np.linalg.lstsq(M_U, t_a, rcond=None)[0]
                except:
                    pass
                    
            # Update V
            M_V = np.einsum('ar,cr->acr', U, W).reshape(nm2 * np2, r)
            for b in range(mp2):
                t_b = np.zeros(nm2 * np2)
                for a in range(nm2):
                    for c in range(np2):
                        t_b[a * np2 + c] = T233[a, b, c]
                try:
                    V[b, :] = np.linalg.lstsq(M_V, t_b, rcond=None)[0]
                except:
                    pass
                    
            # Update W
            M_W = np.einsum('ar,br->abr', U, V).reshape(nm2 * mp2, r)
            for c in range(np2):
                t_c = np.zeros(nm2 * mp2)
                for a in range(nm2):
                    for b in range(mp2):
                        t_c[a * mp2 + b] = T233[a, b, c]
                try:
                    W[c, :] = np.linalg.lstsq(M_W, t_c, rcond=None)[0]
                except:
                    pass
            
            # Check error
            err = 0.0
            for a in range(nm2):
                for b in range(mp2):
                    for c in range(np2):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        target = T233[a, b, c]
                        err += (val - target) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-10:
                break
                
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None

    U233, V233, W233 = find_233_decomposition()
    
    if U233 is not None:
        # Construct 2x3x9 decomposition
        rank = total_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        
        # Block structure
        # B has columns 0..8. Blocks are 0-2, 3-5, 6-8.
        # C has columns 0..8. Blocks are 0-2, 3-5, 6-8.
        for block in range(3):
            for t in range(target_rank_block):
                col = block * target_rank_block + t
                
                # U is same for all blocks
                U[:, col] = U233[:, t]
                
                # V: indices j*p + k. p=9.
                # Block block corresponds to k in 3*block .. 3*block+2
                for j in range(m):
                    for k_local in range(3):
                        k_global = 3 * block + k_local
                        # In 233 decomposition, V index is j*3 + k_local
                        idx_233 = j * 3 + k_local
                        idx_full = j * p + k_global
                        V[idx_full, col] = V233[idx_233, t]
                        
                # W: indices k*n + i. n=2.
                # Block block corresponds to k in 3*block .. 3*block+2
                for k_local in range(3):
                    k_global = 3 * block + k_local
                    for i in range(n):
                        # In 233 decomposition, W index is k_local*2 + i
                        idx_233 = k_local * 2 + i
                        idx_full = k_global * n + i
                        W[idx_full, col] = W233[idx_233, t]
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
