import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    deadline = time.time() + 140
    
    # Try to construct a rank-12 decomposition using separability
    # 2x3 x 3x6 = sum over j of (2x1 x 1x6) for each column j
    # Each 2x1 x 1x6 can be split into three 2x1 x 1x2 blocks (k pairs)
    # Each 2x1 x 1x2 has rank 4 (schoolbook is optimal)
    # Total: 3 j's * 3 k-pairs * 4 = 36? No, that's wrong.
    
    # Let me think more carefully.
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed j, the contribution is: A[:,j] * B[j,:] which is a 2x6 outer product
    # This outer product has rank 1 as a matrix, but as a tensor...
    
    # Actually, let's just try the direct construction for rank 12.
    # If it doesn't work, fall back to search.
    
    # The minimal rank for 2x3x6 is likely 32 (2 * rank(2x3x3)).
    # Search for ranks 30-35 to find a better decomposition than schoolbook (36).
    # We prioritize lower ranks.
    for rank in range(30, 36):
        if time.time() > deadline:
            break
        result = _try_rank_search(rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook 36
    return _schoolbook()

def _try_separable_rank12():
    """Try to construct rank-12 decomposition using separability."""
    n, m, p = 2, 3, 6
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    rank = 12
    U = np.zeros((dimU, rank))
    V = np.zeros((dimV, rank))
    W = np.zeros((dimW, rank))
    
    # For each j in 0..2, and each k_pair in 0..2 (covering k=2*kp, 2*kp+1):
    # We have a 2x1x2 sub-problem
    # The 4 terms for 2x1x2:
    #   term 0: i=0, k_local=0
    #   term 1: i=0, k_local=1
    #   term 2: i=1, k_local=0
    #   term 3: i=1, k_local=1
    # 
    # But wait, 3*3*4 = 36 terms, not 12.
    # The separability gives rank 36, not 12.
    
    # Let me reconsider. Maybe rank 12 is not achievable via this simple separability.
    # Let me try a different approach: use the fact that 2x1x2 has rank 4,
    # but maybe I can share terms across j's.
    
    # Actually, I think the minimal rank might be higher than 12.
    # Let me just do a numerical search for low ranks.
    return None

def _try_rank_search(rank, deadline):
    """Search for a rank-r decomposition using ALS."""
    n, m, p = 2, 3, 6
    dimU = n * m
    dimV = m * p
    dimW = p * n
    
    # Build tensor
    T = np.zeros((dimU, dimV, dimW), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    np.random.seed(rank * 7919 + 42)
    
    # Increase restarts for higher ranks as the problem is smaller in terms of matrix inversion cost
    max_restarts = 100 
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        U = np.random.randn(dimU, rank)
        V = np.random.randn(dimV, rank)
        W = np.random.randn(dimW, rank)
        
        # Normalize
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        residual = np.inf
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Update U
            Z_U = np.einsum('bl,cl->bcl', V, W)
            Z_U_mat = Z_U.transpose(2, 0, 1).reshape(rank, dimV * dimW)
            T_flat = T.reshape(dimU, dimV * dimW)
            U_new, _, _, _ = np.linalg.lstsq(Z_U_mat.T, T_flat.T, rcond=None)
            U = U_new.T
            
            # Update V
            Z_V = np.einsum('al,cl->acl', U, W)
            Z_V_mat = Z_V.transpose(2, 0, 1).reshape(rank, dimU * dimW)
            T_flat_V = T.transpose(1, 0, 2).reshape(dimV, dimU * dimW)
            V_new, _, _, _ = np.linalg.lstsq(Z_V_mat.T, T_flat_V.T, rcond=None)
            V = V_new.T
            
            # Update W
            Z_W = np.einsum('al,bl->abl', U, V)
            Z_W_mat = Z_W.transpose(2, 0, 1).reshape(rank, dimU * dimV)
            T_flat_W = T.transpose(2, 0, 1).reshape(dimW, dimU * dimV)
            W_new, _, _, _ = np.linalg.lstsq(Z_W_mat.T, T_flat_W.T, rcond=None)
            W = W_new.T
            
            # Check residual
            T_decomp = np.einsum('al,bl,cl->abc', U, V, W)
            residual = np.linalg.norm(T - T_decomp)
            if residual < 1e-12:
                break
        
        if residual < 1e-10:
            result = _snap_to_rationals(U, V, W, T)
            if result is not None:
                return result
    
    return None

def _snap_to_rationals(U, V, W, T):
    """Try to snap to integer/half-integer values."""
    def check_and_snap(mat, tol=1e-5):
        # Try rounding to nearest 0.5
        snapped = np.round(mat * 2) / 2
        if np.allclose(mat, snapped, atol=tol):
            return snapped
        # Try rounding to nearest integer
        snapped_int = np.round(mat)
        if np.allclose(mat, snapped_int, atol=tol):
            return snapped_int
        return None
    
    U_s = check_and_snap(U)
    V_s = check_and_snap(V)
    W_s = check_and_snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        T_check = np.einsum('al,bl,cl->abc', U_s, V_s, W_s)
        if np.allclose(T_check, T, atol=1e-5):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
            
    return None

def _schoolbook():
    n, m, p = 2, 3, 6
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
