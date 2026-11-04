import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(R, T, atol=tol)

    def snap_to_half_integers(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def verify_and_return(U, V, W):
        # Try direct
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Try snapped
        U_r, V_r, W_r = snap_to_half_integers(U, V, W)
        if check_decomposition(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    def get_schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def run_als(U, V, W, T, max_iter=100, tol=1e-10):
        """
        Run Alternating Least Squares to refine U, V, W to fit T.
        U: (size_u, rank), V: (size_v, rank), W: (size_w, rank)
        """
        size_u, rank = U.shape
        size_v, _ = V.shape
        size_w, _ = W.shape
        
        # Precompute reshaped T for speed
        T_1 = T.reshape(size_u, size_v * size_w)
        T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
        T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # Minimize || T - sum_r U[:,r] outer V[:,r] outer W[:,r] ||
            # For fixed V, W, U[:,r] is least squares solution to:
            # T_1 * (V[:,r] * W[:,r])^T = U[:,r]
            # Matrix M_vw has columns v_r * w_r
            M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
            # Solve M_vw.T @ U.T = T_1.T
            # U.T = M_vw @ (M_vw.T @ M_vw)^-1 ... use lstsq
            U_new, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
            U = U_new.T
            
            # Update V
            M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
            V_new, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
            V = V_new.T
            
            # Update W
            M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
            W_new, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
            W = W_new.T
            
            # Compute error
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(R - T)
            
            if err < tol:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
            
        return U, V, W, err

    def reduce_rank(U, V, W, target_rank, T):
        """
        Try to reduce the rank of the decomposition by dropping columns and refining.
        """
        current_rank = U.shape[1]
        if current_rank == target_rank:
            return U, V, W
        
        # We need to drop (current_rank - target_rank) columns.
        # Strategy: Drop columns with smallest norm, then refine.
        # Try multiple strategies
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        num_drops = current_rank - target_rank
        
        # Strategy 1: Drop min norm columns iteratively
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        for _ in range(num_drops):
            if time.time() > deadline - 10:
                break
            norms_u = np.linalg.norm(U_cur, axis=0)
            norms_v = np.linalg.norm(V_cur, axis=0)
            norms_w = np.linalg.norm(W_cur, axis=0)
            # Combine norms to find "least important" term
            # A term r is important if U[:,r], V[:,r], W[:,r] are large
            # Approximate importance by product of norms or sum
            importances = norms_u * norms_v * norms_w
            idx_to_drop = np.argmin(importances)
            
            # Remove column
            mask = np.arange(current_rank) != idx_to_drop
            U_cur = U_cur[:, mask]
            V_cur = V_cur[:, mask]
            W_cur = W_cur[:, mask]
            current_rank -= 1
            
        # Refine the reduced set
        U_ref, V_ref, W_ref, err = run_als(U_cur, V_cur, W_cur, T, max_iter=200)
        
        if err < best_err:
            best_U, best_V, best_W = U_ref, V_ref, W_ref
            best_err = err

        # Strategy 2: Random drop and refine
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            mask = np.random.choice(current_rank, target_rank, replace=False)
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            U_ref, V_ref, W_ref, err = run_als(U_sub, V_sub, W_sub, T, max_iter=200)
            
            if err < best_err:
                best_U, best_V, best_W = U_ref, V_ref, W_ref
                best_err = err
                
        return best_U, best_V, best_W

    # 1. Start with Schoolbook (Rank 27)
    U, V, W = get_schoolbook_decomposition()
    
    # 2. Try to reduce rank
    # We know rank is 23. Try 26, 25, 24, 23.
    # If we fail at 26, we can still return 27? No, we want the best.
    # But if we fail to reduce, we return the current best found.
    
    best_result = None
    
    # Try to find a valid solution at each rank
    # Start from 26 down to 23
    # Note: The current U,V,W is rank 27.
    
    for target_rank in [26, 25, 24, 23]:
        if time.time() > deadline - 20:
            break
            
        U_red, V_red, W_red = reduce_rank(U, V, W, target_rank, T)
        
        if U_red is not None:
            # Check if this reduced decomposition is exact (or snap-able)
            res = verify_and_return(U_red, V_red, W_red)
            if res is not None:
                best_result = res
                break # Found a lower rank solution, stop
            
            # Even if not snapped, keep it as the current best candidate for further reduction?
            # No, if it's not exact, it's not a valid decomposition for the task (which requires exactness).
            # However, we can use it as a warm start for the next lower rank?
            # If it's not exact, it's a poor start.
            # But if the error is small, it might be close.
            # Let's update U,V,W to the refined version regardless of exactness, 
            # but only return if exact.
            U, V, W = U_red, V_red, W_red

    # If we didn't find a lower rank solution, return schoolbook
    if best_result is None:
        # Verify schoolbook just in case
        res = verify_and_return(U, V, W) # This is now potentially reduced but inexact
        # We should return the original schoolbook if the reduced ones failed verification
        U_school, V_school, W_school = get_schoolbook_decomposition()
        res = verify_and_return(U_school, V_school, W_school)
        if res:
            best_result = res
        else:
            # Fallback to manual construction of schoolbook lists
            rank = n * m * p
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_list[i * m + j][column] = 1
                        V_list[j * p + k][column] = 1
                        W_list[k * n + i][column] = 1
                        column += 1
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    return best_result
