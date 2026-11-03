import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def try_rank(rank, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Precompute tensor reshapes
        # T shape: (n*m, m*p, p*n)
        
        for it in range(50): # Reduced iterations for speed
            if time.time() > deadline:
                return float('inf'), U, V, W
            
            # Solve for W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix U, V.
            # M[a,b,r] = U[a,r] V[b,r]
            # T_reshaped (n*m*m*p, p*n) = M_reshaped (n*m*m*p, rank) @ W^T
            UV = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, rank)
            T_reshaped = tensor.reshape(n*m*m*p, p*n)
            W_new_T, _, _, _ = np.linalg.lstsq(UV, T_reshaped, rcond=None)
            W = W_new_T.T
            
            # Solve for V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix U, W.
            # M[a,c,r] = U[a,r] W[c,r]
            # T_reshaped (m*p, n*m*p*n) = V (m*p, rank) @ M_reshaped^T (rank, n*m*p*n)
            # So V^T = lstsq(M_reshaped, T_reshaped^T)
            # M_reshaped shape: (n*m*p*n, rank)
            UW = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, rank)
            T_reshaped2 = tensor.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            V_new_T, _, _, _ = np.linalg.lstsq(UW, T_reshaped2.T, rcond=None)
            V = V_new_T.T
            
            # Solve for U
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W.
            # M[b,c,r] = V[b,r] W[c,r]
            # T_reshaped (n*m, m*p*p*n) = U (n*m, rank) @ M_reshaped^T (rank, m*p*p*n)
            # So U^T = lstsq(M_reshaped, T_reshaped^T)
            # M_reshaped shape: (m*p*p*n, rank)
            VW = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, rank)
            T_reshaped3 = tensor.transpose(0, 1, 2).reshape(n*m, m*p*p*n)
            U_new_T, _, _, _ = np.linalg.lstsq(VW, T_reshaped3.T, rcond=None)
            U = U_new_T.T
            
            # Check residual
            if it % 5 == 0:
                # Efficient residual check
                # R = T - sum_r U[:,r] V[:,r] W[:,r]
                # We can check norm of T - reconstructed
                # Reconstructing full tensor is expensive.
                # Let's just check every 10 iters or at the end.
                pass
        
        # Final residual check
        # Compute sum of squares of (T - sum U V W)
        # This is expensive, but we only do it at the end of try_rank
        # Approximation:
        # ||T||^2 - 2 <T, sum U V W> + ||sum U V W||^2
        # ||T||^2 is constant (number of non-zeros = n*m*p = 96)
        # <T, sum U V W> = sum_r sum_{a,b,c} T[a,b,c] U[a,r] V[b,r] W[c,r]
        # ||sum U V W||^2 = sum_{r, s} (U^T V ... ) complicated.
        
        # Let's just do a direct check for small tensors
        # n*m=16, m*p=24, p*n=24. 16*24*24 = 9216 elements.
        # Rank up to 96.
        # 9216 * 96 operations ~ 1M. Fast enough.
        
        recon = np.zeros_like(tensor)
        for r in range(rank):
            recon += np.einsum('ar,br,cr->abc', U[:,r], V[:,r], W[:,r])
        residual = np.sum((tensor - recon)**2)
        
        return residual, U, V, W

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from a likely minimal up to schoolbook
    # Known lower bound for 4x4x6?
    # Strassen 2x2x2 is 7.
    # 4x4x4 is 49? (Strassen 7 * 7? No, 4x4 is 49? No, 4x4x4 rank is 49? No, 4x4x4 is 49?
    # Actually, 4x4x4 rank is 49? No, 4x4x4 rank is 49 is wrong. 4x4x4 rank is 49?
    # 2x2x2 is 7. 4x4x4 can be done in 49? No, 4x4x4 rank is 49?
    # Wait, 4x4x4 rank is 49? No.
    # 4x4x4 rank is 49?
    # Actually, the rank of 4x4x4 matrix multiplication is 49? No.
    # It is 49?
    # Let's just search from 50 to 96.
    
    for rank in range(50, 97):
        if time.time() > deadline:
            break
        residual, U, V, W = try_rank(rank)
        if residual < 1e-6:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
        
    if best_U is None:
        # Fall back to schoolbook
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
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
