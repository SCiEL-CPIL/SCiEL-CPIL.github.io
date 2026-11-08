import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # Build the target tensor T
    # T[i, j, k] is 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # The tensor is defined such that C = A B corresponds to T
    # T indices: (i, j, k) where i in 0..n-1, j in 0..m-1, k in 0..p-1
    # The decomposition is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Note: The problem statement says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This matches the index mapping:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        # Reconstruct
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # T_recon[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
        # Use einsum
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    def check_half_int(M, tol=1e-6):
        for val in M.flatten():
            if abs(2*val - round(2*val)) > tol:
                return False
        return True

    def round_half_int(M):
        return np.round(2*M) / 2.0

    def try_rank(r, max_iter=100, seed=0):
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to prevent explosion
        norm = np.sqrt(r)
        U /= norm
        V /= norm
        W /= norm
        
        best_err = float('inf')
        
        for it in range(max_iter):
            # Update U
            # T_flat[i, (j,k)] = sum_c U[i,c] * V[j,c] * W[k,c]
            # Let M_u[c, (j,k)] = V[j,c] * W[k,c]
            # T_flat = U @ M_u.T
            # U = T_flat @ M_u.T @ (M_u @ M_u.T)^-1
            M_u = np.einsum('jc,kc->jkc', V, W).reshape(r, -1) # (r, mp*pn)
            T_flat = T.reshape(n*m, -1) # (nm, mp*pn)
            # Solve U @ M_u.T = T_flat.T => U = T_flat.T @ (M_u.T)^+
            # Using least squares: M_u.T @ U.T = T_flat.T
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_u.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V
            # T_flat[j, (i,k)] = sum_c V[j,c] * U[i,c] * W[k,c]
            # M_v[c, (i,k)] = U[i,c] * W[k,c]
            M_v = np.einsum('ic,kc->ick', U, W).reshape(r, -1) # (r, nm*pn)
            T_flat_v = T.transpose(1, 0, 2).reshape(m*p, -1) # (mp, nm*pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W
            # T_flat[k, (i,j)] = sum_c W[k,c] * U[i,c] * V[j,c]
            # M_w[c, (i,j)] = U[i,c] * V[j,c]
            M_w = np.einsum('ic,jc->ijc', U, V).reshape(r, -1) # (r, nm*mp)
            T_flat_w = T.transpose(2, 0, 1).reshape(p*n, -1) # (pn, nm*mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Calculate error
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    # Try rounding
                    U_r = round_half_int(U)
                    V_r = round_half_int(V)
                    W_r = round_half_int(W)
                    if verify(U_r, V_r, W_r, tol=1e-5):
                        return U_r, V_r, W_r
                    # Also try rounding to integers? The problem says integer/half-integer.
                    # round_half_int covers both (integers are multiples of 0.5)
        
        return None

    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    def get_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_s, V_s, W_s = get_schoolbook()
    best_result = (U_s, V_s, W_s)
    best_rank = n * m * p

    # Try to find lower ranks
    # We iterate from a lower bound upwards.
    # Theoretical lower bound for rank of M_{n,m,p} is not trivial, but 40 is the schoolbook.
    # Let's try 30 to 39.
    # If we find a valid one, we can try to go lower?
    # It's safer to go from 40 downwards? No, if we find 35, we should check 34, 33...
    # But searching 30-40 might take time.
    # Let's try 35, 36, 37, 38, 39.
    # If found, check lower.
    
    ranks_to_try = list(range(39, 29, -1)) # 39 down to 30
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        found = False
        for seed in range(10): # 10 restarts per rank
            if time.time() > deadline:
                break
                
            res = try_rank(r, max_iter=50, seed=seed)
            if res is not None:
                U, V, W = res
                # Verify once more
                if verify(U, V, W, tol=1e-6) and check_half_int(U) and check_half_int(V) and check_half_int(W):
                    best_rank = r
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            # We found a valid rank r. Can we go lower?
            # The loop continues to lower r (since we are iterating downwards in the list? No, range(39, 29, -1) is 39, 38, ... 30)
            # So if we found 35, we will try 34, 33...
            # This is good.
            continue
        else:
            # If not found for this r, maybe try higher? No, we are going down.
            # If we fail at 39, we try 38.
            # This assumes that if a rank r is possible, it's likely r+1 is possible?
            # Not necessarily, but usually low rank decompositions are harder to find.
            # However, if we can't find 39, we keep best_result as 40 (or whatever was found previously).
            # Since we start with 40, best_result is always valid.
            pass

    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
