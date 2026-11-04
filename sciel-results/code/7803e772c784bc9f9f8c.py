import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin

    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Check exact decomposition
        # T[a, b, c] = sum_r U_r[a, r] V_r[b, r] W_r[c, r]
        # Vectorized check:
        # Reshape T to (nm, mp*pn)
        T_flat = T.reshape(nm, -1)
        
        # Construct A for U: (mp*pn, rank)
        # A[idx, r] = V_r[b, r] * W_r[c, r]
        # V_r is (mp, rank), W_r is (pn, rank)
        A_U = (V_r[:, None, :] * W_r[None, :, :]).reshape(-1, U_r.shape[1])
        
        # Check if U_r @ A_U.T is close to T_flat
        # U_r is (nm, rank), A_U.T is (rank, mp*pn)
        # Result is (nm, mp*pn)
        Rec_U = U_r @ A_U.T
        if not np.allclose(Rec_U, T_flat, atol=1e-6):
            return False
            
        # Construct A for V: (nm*pn, rank)
        # A_V[idx, r] = U_r[a, r] * W_r[c, r]
        A_V = (U_r[:, None, :] * W_r[None, :, :]).reshape(-1, V_r.shape[1])
        T_flat_V = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, -1) # Shape (mp, nm*pn)
        Rec_V = V_r @ A_V.T
        if not np.allclose(Rec_V, T_flat_V, atol=1e-6):
            return False
            
        # Construct A for W: (nm*mp, rank)
        # A_W[idx, r] = U_r[a, r] * V_r[b, r]
        A_W = (U_r[:, None, :] * V_r[None, :, :]).reshape(-1, W_r.shape[1])
        T_flat_W = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, -1) # Shape (pn, nm*mp)
        Rec_W = W_r @ A_W.T
        if not np.allclose(Rec_W, T_flat_W, atol=1e-6):
            return False
            
        return True

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

    fallback = schoolbook()

    def try_als(rank, max_iters=100, num_restarts=5):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            # Initialize with small values to help convergence to integers?
            # Or random?
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Pre-compute T flattens
            T_flat_U = T.reshape(nm, -1) # (nm, mp*pn)
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, -1) # (mp, nm*pn)
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, -1) # (pn, nm*mp)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # A_U: (mp*pn, rank)
                A_U = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                # Solve U @ A_U.T = T_flat_U
                # A_U.T is (rank, mp*pn)
                U, residuals, rank_val, sv = np.linalg.lstsq(A_U.T, T_flat_U, rcond=None)
                
                # Update V
                # A_V: (nm*pn, rank)
                A_V = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                # Solve V @ A_V.T = T_flat_V
                V, residuals, rank_val, sv = np.linalg.lstsq(A_V.T, T_flat_V, rcond=None)
                
                # Update W
                # A_W: (nm*mp, rank)
                A_W = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                # Solve W @ A_W.T = T_flat_W
                W, residuals, rank_val, sv = np.linalg.lstsq(A_W.T, T_flat_W, rcond=None)
                
                # Check convergence / validity
                if it % 5 == 0:
                    if verify(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    best_result = None
    # Try ranks from low to high
    # Start from a reasonable lower bound.
    # 4x4x4 is 49.
    # 4x5x6 is likely close to 120.
    # Try 119, 118... down to 100?
    # Or 60 to 120?
    # With the speedup, we can try more.
    for r in range(60, 120):
        if time.time() > deadline - 5:
            break
            
        result = try_als(r, max_iters=100, num_restarts=3)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = fallback
        
    return best_result
