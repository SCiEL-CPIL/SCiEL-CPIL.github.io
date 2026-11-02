"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the tensor indices
    # T[a, b, c] = 1 if a%m == b//p and (b%p)*n + a//m == c
    # We store the non-zero entries as a list of (a, b, c)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c))
                
    def check_solution(U, V, W):
        # U, V, W are numpy arrays of shape (nm, r), (mp, r), (pn, r)
        # Check if sum_c U[a,c]*V[b,c]*W[c,c] == 1 for all non_zeros
        # and 0 otherwise (though we only care about non_zeros being 1, 
        # and the problem implies the tensor is exactly the matmul tensor, 
        # so other entries must be 0).
        # However, the tensor is sparse. We only need to check the non-zero entries 
        # sum to 1, and implicitly the rest are 0 if the rank is low enough? 
        # No, we must check that the reconstruction matches the tensor exactly.
        # Since the tensor has 1s at non_zeros and 0 elsewhere, 
        # we must ensure the reconstructed tensor matches.
        
        r = U.shape[1]
        # Efficient check:
        # Construct the full tensor? No, too big (6*21*14 = 1764). It's small enough.
        T_recon = np.zeros((nm, mp, pn))
        # Use einsum or loops? Loops are fine for small r.
        for c in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
        # Check non-zero entries
        for a, b, c_idx in non_zeros:
            if not np.isclose(T_recon[a, b, c_idx], 1.0, atol=1e-9):
                return False
        # Check zero entries? 
        # The tensor for matrix multiplication is 1 at the specific indices and 0 elsewhere.
        # If the rank is less than nm*mp*pn, it's possible that some "zero" entries become non-zero.
        # We must check that all entries are either 0 or 1 (or close to them) and match the pattern.
        # Actually, just checking the non-zeros is not enough. We need to ensure no "leakage".
        # Let's check the Frobenius norm of the difference.
        T_true = np.zeros((nm, mp, pn))
        for a, b, c_idx in non_zeros:
            T_true[a, b, c_idx] = 1.0
            
        if not np.allclose(T_recon, T_true, atol=1e-9):
            return False
            
        return True

    def round_to_half(X):
        return np.round(X * 2) / 2

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank decomposition
    # Start from 38 up to 41
    for rank in range(38, 42):
        # Try a few random initializations
        for trial in range(10):
            np.random.seed(trial * 100 + rank)
            
            # Initialize with random values
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale down to avoid overflow/instability initially?
            # Or use a structured init?
            # Let's try a simple ALS loop
            
            success = False
            start_time = time.time()
            
            for it in range(100): # Max iterations per trial
                if time.time() - start_time > 10: # 10s per trial
                    break
                
                # Update U
                # T[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
                # For fixed V, W, we want to find U such that 
                # U @ (V * W) approx T
                # Reshape T to (nm, mp*pn)
                T_mat = np.zeros((nm, mp * pn))
                for a, b, c_idx in non_zeros:
                    T_mat[a, b * pn + c_idx] = 1.0
                
                # Construct matrix M where M[:, b*pn+c] = V[b] * W[c]
                # M is (rank, mp*pn)
                # We want U @ M.T = T_mat
                # So U = T_mat @ M.T @ (M @ M.T)^-1  (Least Squares)
                
                # Efficiently construct M.T
                # M.T has shape (mp*pn, rank)
                # M.T[ b*pn+c, k ] = V[b,k] * W[c,k]
                MT = np.zeros((mp * pn, rank))
                for b in range(mp):
                    for c in range(pn):
                        MT[b * pn + c, :] = V[b, :] * W[c, :]
                
                # Solve U * MT.T = T_mat  => U = T_mat * (MT.T * MT)^-1 * MT
                # Or use np.linalg.lstsq
                try:
                    U_new, _, _, _ = np.linalg.lstsq(MT.T, T_mat, rcond=None)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                # T[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
                # For fixed U, W, 
                # V @ (U * W) approx T (reshaped to mp, nm*pn)
                T_mat_V = np.zeros((mp, nm * pn))
                for a, b, c_idx in non_zeros:
                    T_mat_V[b, a * pn + c_idx] = 1.0
                
                MT_V = np.zeros((nm * pn, rank))
                for a in range(nm):
                    for c in range(pn):
                        MT_V[a * pn + c, :] = U[a, :] * W[c, :]
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(MT_V.T, T_mat_V, rcond=None)
                except np.linalg.LinAlgError:
                    break
                
                # Update W
                # T[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
                # For fixed U, V,
                # W @ (U * V) approx T (reshaped to pn, nm*mp)
                T_mat_W = np.zeros((pn, nm * mp))
                for a, b, c_idx in non_zeros:
                    T_mat_W[c_idx, a * mp + b] = 1.0
                
                MT_W = np.zeros((nm * mp, rank))
                for a in range(nm):
                    for b in range(mp):
                        MT_W[a * mp + b, :] = U[a, :] * V[b, :]
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(MT_W.T, T_mat_W, rcond=None)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence / validity
                if check_solution(U, V, W):
                    # Round to half-integers and check again
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    if check_solution(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    # If rounding fails, try to keep floating point? 
                    # The problem asks for integer/half-integer entries.
                    # If the floating point solution is very close to half-integers, 
                    # it should pass.
            
            # If we exit the loop without success, try next trial

    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
