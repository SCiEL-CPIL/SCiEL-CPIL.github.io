import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct the trivial rank n*m*p solution efficiently
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    R_trivial = n * m * p
    U_trivial = np.zeros((n * m, R_trivial))
    V_trivial = np.zeros((m * p, R_trivial))
    W_trivial = np.zeros((p * n, R_trivial))
    
    # We can fill these using broadcasting or simple loops, but loops are fine for 42
    # To be safe and fast, we'll use the loop but it's negligible compared to optimization
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, idx] = 1.0
                V_trivial[j * p + k, idx] = 1.0
                W_trivial[k * n + i, idx] = 1.0
                idx += 1
                
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = R_trivial

    # Helper to verify a decomposition
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        try:
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if not np.allclose(S, 1.0, atol=tol):
                return False
        except Exception:
            return False
            
        return True

    def try_find_decomp_fast(n, m, p, target_rank, time_limit):
        """
        Attempts to find a rank target_rank decomposition using fast gradient descent.
        """
        if time.time() > time_limit:
            return None
            
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        # We will optimize U, V, W separately to avoid large vectorization issues in least_squares
        # But gradient descent on separate tensors is tricky due to non-convexity.
        # Let's stick to a simple random restart with a few gradient steps.
        
        def compute_residual(U, V, W):
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            return S - 1.0

        max_restarts = 20
        steps_per_restart = 50
        lr = 0.01
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize
            U = np.random.randn(n, m, target_rank) * 0.1
            V = np.random.randn(m, p, target_rank) * 0.1
            W = np.random.randn(p, n, target_rank) * 0.1
            
            for step in range(steps_per_restart):
                if time.time() > time_limit:
                    break
                
                # Compute gradient
                # S = einsum('ijr,jkr,kir->ijk', U, V, W)
                # dU = einsum('jkr,kir,ijk->ijr', V, W, S-1)
                # This is approximate gradient for sum (S-1)^2
                
                S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                Res = S - 1.0
                
                # Gradients
                dU = np.einsum('jkr,kir,ijk->ijr', V, W, Res)
                dV = np.einsum('ijr,kir,ijk->jkr', U, W, Res)
                dW = np.einsum('ijr,jkr,ijk->kir', U, V, Res)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # Check residual
            S_final = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            if np.allclose(S_final, 1.0, atol=1e-6):
                # Snap to half-integers
                U_flat = U.reshape(n*m, target_rank)
                V_flat = V.reshape(m*p, target_rank)
                W_flat = W.reshape(p*n, target_rank)
                
                U_half = np.round(U_flat * 2) / 2
                V_half = np.round(V_flat * 2) / 2
                W_half = np.round(W_flat * 2) / 2
                
                if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-5):
                    return U_half, V_half, W_half
                    
        return None

    # Strategy: Try to find rank < 42.
    # Given the difficulty, we only try a few ranks close to 42 with a tight budget.
    # The lower bound is 21. But 42 is likely the answer.
    # We will try rank 41 and 40 quickly.
    
    ranks_to_try = [41, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        res = try_find_decomp_fast(n, m, p, r, deadline - 10)
        if res:
            U_new, V_new, W_new = res
            best_U = U_new
            best_V = V_new
            best_W = W_new
            best_rank = r
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
