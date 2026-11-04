import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin from 150s limit
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i for some i,j,k
    # Else 0
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        # Compute sum_r U[:, r] V[:, r]^T W[:, r]^T? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute the reconstructed tensor and compare.
        # For efficiency, we can check a subset or use the full tensor.
        # Full tensor is 27x27x27 = 19683 elements.
        # Reconstruction:
        # R = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow in Python if done naively, but numpy einsum is fast.
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False, None, None, None
        
        # Check if entries are integer/half-integer
        # The problem says "integer/half-integer entries".
        # We should round U, V, W to nearest half-integer and re-check.
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True, U_r, V_r, W_r
        
        # Also try integer rounding, just in case
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        R_i = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        if np.allclose(R_i, T, atol=1e-9):
            return True, U_i, V_i, W_i
            
        return False, None, None, None

    def find_decomposition(rank, max_attempts=10, max_iter=50):
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
                
            # Random initialization
            # Scale initialization to help convergence
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(size_u, rank) * scale
            V = np.random.randn(size_v, rank) * scale
            W = np.random.randn(size_w, rank) * scale
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                    
                # Update U
                # T_unfolded_1 shape (size_u, size_v * size_w)
                # M_vw shape (size_v * size_w, rank)
                # M_vw[ (b * size_w + c), r ] = V[b, r] * W[c, r]
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                try:
                    U, residuals, rank_u, s = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                    U = U.T
                except np.linalg.LinAlgError:
                    break
                    
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                try:
                    V, residuals, rank_v, s = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                    V = V.T
                except np.linalg.LinAlgError:
                    break
                    
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                try:
                    W, residuals, rank_w, s = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                    W = W.T
                except np.linalg.LinAlgError:
                    break
                    
                # Check convergence periodically
                if it % 5 == 0 or it == max_iter - 1:
                    is_valid, U_r, V_r, W_r = check_decomposition(U, V, W)
                    if is_valid:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        return None

    # Fallback to schoolbook
    def get_schoolbook():
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = None
    
    # Try ranks from 20 to 26
    # Lower ranks are harder, so we give them more attempts? 
    # Or we try them in order and stop if found.
    # Rank 20 is the theoretical best. Rank 27 is schoolbook.
    
    # We have ~140 seconds.
    # Let's allocate time based on rank difficulty.
    # Ranks 20-22: 20 attempts, 50 iters
    # Ranks 23-26: 10 attempts, 30 iters
    
    for rank in range(20, 27):
        if time.time() > deadline - 5: # Leave 5s for cleanup/return
            break
            
        # Adjust attempts based on remaining time
        remaining = deadline - time.time()
        if rank <= 22:
            max_attempts = int(remaining / 5) # ~5s per attempt
            max_iter = 50
            if max_attempts < 1:
                continue
        else:
            max_attempts = int(remaining / 10)
            max_iter = 30
            if max_attempts < 1:
                continue
        
        res = find_decomposition(rank, max_attempts=max_attempts, max_iter=max_iter)
        if res is not None:
            best_result = res
            break # Found minimal rank, stop

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()
