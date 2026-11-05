# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Build target tensor for verification
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        result = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
        return np.allclose(result, target, atol=tol)

    # 2. Structured search for rank reduction using Gradient Descent
    # Try to find a decomposition of rank R < base_rank
    
    np.random.seed(42)
    
    # Function to attempt optimization for a given rank
    def try_optimize_rank(R, max_iters=500, lr=0.01, restarts=3):
        for r in range(restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Simple Gradient Descent
            # Residual R_ijk = sum_c U_ic V_jc W_kc - T_ijk
            # d/dU_ic = sum_jk V_jc W_kc R_ijk
            # d/dV_jc = sum_ik U_ic W_kc R_ijk
            # d/dW_kc = sum_ij U_ic V_jc R_ijk
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Compute current tensor
                T_curr = np.einsum('ic,jc,kc->ijk', U, V, W)
                Res = T_curr - target
                
                # Compute gradients
                dU = np.einsum('ijk,jc,kc->ic', Res, V, W)
                dV = np.einsum('ijk,ic,kc->jc', Res, U, W)
                dW = np.einsum('ijk,ic,jc->kc', Res, U, V)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if np.linalg.norm(Res) < 1e-6:
                    # Snap to integers/half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_decomp(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # If not converged, try again with different init
        return None

    # Try ranks from base_rank - 1 downwards
    for R in range(base_rank - 1, max(base_rank - 5, 0), -1):
        if time.time() > deadline - 10:
            break
            
        result = try_optimize_rank(R)
        if result:
            best = result
            best_rank = R
            break
            
    return best
# EVOLVE-BLOCK-END
