import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 3, 5, 5
    size_U = n * m  # 15
    size_V = m * p  # 25
    size_W = p * n  # 15
    
    # Precompute target tensor T[I, J, K]
    # I = i*m + j, J = j*p + k, K = k*n + i
    T_target = np.zeros((size_U, size_V, size_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T_target[I, J, K] = 1.0

    T_flat = T_target.flatten()
    
    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        # Res = np.einsum('ic,jc,kc->ijk', U, V, W)
        # Faster: matmul
        # U: (15, R), V: (25, R), W: (15, R)
        # Res[I, J, K] = sum_c U[I,c] V[J,c] W[K,c]
        # We can compute this as:
        # For each c, outer(U[:,c], V[:,c]) * W[:,c]
        # Or use einsum
        Res = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        err = np.linalg.norm(Res - T_target)
        return err < tol, err

    def snap_to_halfint(M):
        return np.round(M * 2) / 2

    def residual(vec):
        R = len(vec) // (size_U + size_V + size_W)
        U_flat = vec[:size_U * R]
        V_flat = vec[size_U * R : size_U * R + size_V * R]
        W_flat = vec[size_U * R + size_V * R : ]
        
        U = U_flat.reshape(size_U, R)
        V = V_flat.reshape(size_V, R)
        W = W_flat.reshape(size_W, R)
        
        Res = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        return (Res - T_target).flatten()

    def try_rank(R, num_restarts=10):
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None, None, None
            
            # Initialization
            if restart % 2 == 0:
                # Random
                U_init = np.random.randn(size_U, R) * 0.5
                V_init = np.random.randn(size_V, R) * 0.5
                W_init = np.random.randn(size_W, R) * 0.5
            else:
                # Structured: Use schoolbook basis projected
                # Create schoolbook (rank 75)
                U_school = np.zeros((size_U, 75))
                V_school = np.zeros((size_V, 75))
                W_school = np.zeros((size_W, 75))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i * m + j, col] = 1
                            V_school[j * p + k, col] = 1
                            W_school[k * n + i, col] = 1
                            col += 1
                            
                # Random linear combination of schoolbook terms
                # This ensures the initial guess lies in the span of the schoolbook decomposition
                # which is the correct manifold
                coeffs = np.random.randn(75, R)
                U_init = U_school @ coeffs
                V_init = V_school @ coeffs
                W_init = W_school @ coeffs
                
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares
                res = scipy.optimize.least_squares(residual, x0, method='trf', 
                                                    ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                                    max_nfev=500)
                
                if res.cost < 1e-8:
                    U = res.x[:size_U * R].reshape(size_U, R)
                    V = res.x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
                    W = res.x[size_U * R + size_V * R : ].reshape(size_W, R)
                    
                    # Snap to half-integers
                    U_snap = snap_to_halfint(U)
                    V_snap = snap_to_halfint(V)
                    W_snap = snap_to_halfint(W)
                    
                    valid, err = verify(U_snap, V_snap, W_snap)
                    if valid:
                        return U_snap, V_snap, W_snap
                        
            except Exception as e:
                pass
                
        return None, None, None

    best_rank = 75
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_fb = np.zeros((size_U, 75))
    V_fb = np.zeros((size_V, 75))
    W_fb = np.zeros((size_W, 75))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_fb, V_fb, W_fb

    # Search for lower ranks
    # Based on small matrix multiplication, rank is likely < 75
    # Try a range. 3x5x5 might be around 50-65.
    # We will try from 55 upwards to 74.
    # To save time, we will prioritize lower ranks.
    
    for R in range(55, 75):
        if time.time() > deadline - 20:
            break
            
        # More restarts for lower ranks? Or fewer?
        # Lower ranks are harder to find, so maybe more restarts.
        # But time is limited.
        num_restarts = 20 if R < 60 else 10
        
        U_sol, V_sol, W_sol = try_rank(R, num_restarts)
        
        if U_sol is not None:
            best_rank = R
            best_U = U_sol
            best_V = V_sol
            best_W = W_sol
            break # Found a valid one, stop? 
            # The task asks for rank-minimal. If we find one at R, 
            # we should check if R-1 is possible? 
            # Given time constraints, finding one is a big improvement.
            # We can try to go lower if time permits.
            # But let's just return the first valid one found in ascending order.
            # Wait, the loop is ascending. So the first one found is the minimum in this range.
            # We can break.

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
