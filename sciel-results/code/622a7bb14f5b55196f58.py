import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
    # Build the tensor T
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 115.0 # Safety margin for 120s limit
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    def check_solution(U, V, W, tol=1e-6):
        # Check if rounding to half-integers yields an exact decomposition
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        
        return False, None, None, None

    # Pre-compute matricizations for speed
    # T_c has shape (N*M, P)
    T_c = T.reshape(N * M, P)
    # T_a has shape (N, M*P)
    T_a = T.reshape(N, M * P)
    # T_b has shape (M, N*P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)

    # Search ranks from 35 down to 20
    # We prioritize finding a valid solution quickly.
    # Ranks 35, 34, 33 are very likely to be solvable.
    # Lower ranks are harder.
    ranks_to_try = list(range(35, 19, -1))
    
    for R in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Number of attempts: more for higher ranks initially to secure a good baseline,
        # but we want to go lower. Let's do 5 attempts per rank.
        attempts = 5
        
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(1000 + R * 100 + attempt)
            # Initialize with small random values to avoid immediate saturation
            scale = 0.5
            U = np.random.randn(N, R) * scale
            V = np.random.randn(M, R) * scale
            W = np.random.randn(P, R) * scale
            
            converged = False
            max_iters = 100 # Increased iterations, but guarded by time
            
            for it in range(max_iters):
                if time.time() - start_time > timeout:
                    break
                
                # Update W: Minimize || T_c - (U * V) W^T ||_F
                # Let A = U * V (element-wise product reshaped)
                # A[i, r] = U[i, r] * V[j, r] where i = a*m+b
                # This is not a simple matrix product.
                # T_c[a*m+b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed r, the contribution to T_c is W[:, r] * (U[:, r] * V[:, r])^T
                # So T_c = sum_r W[:, r] (U[:, r] * V[:, r])^T
                # Let E_W be matrix of shape (N*M, R) where E_W[a*m+b, r] = U[a, r] * V[b, r]
                # Then T_c = E_W @ W.T
                # So W = (E_W.T @ E_W)^-1 E_W.T @ T_c.T
                # Using lstsq: W.T = lstsq(E_W, T_c.T)
                
                # Construct E_W
                # U is (N, R), V is (M, R)
                # We need U[a, r] * V[b, r] for all a, b, r
                # This is a Khatri-Rao product or similar.
                # Efficient way: 
                # Reshape U to (n, m, R) and V to (m, p, R)? No, indices are mixed.
                # a = i*m+j, b = j*p+k.
                # It is easier to just compute the matrix E_W directly using broadcasting if possible, 
                # or loop over r. R is small (<36), N*M is 36.
                
                # Let's compute E_W
                # E_W[i, r] where i corresponds to (a, b)
                # We can construct a matrix B of shape (N, M, R) such that B[a, b, r] = U[a, r] * V[b, r]
                # Then reshape B to (N*M, R)
                
                # Using broadcasting:
                # U_expanded: (N, 1, R)
                # V_expanded: (1, M, R)
                # Product: (N, M, R)
                E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
                
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U: Minimize || T_a - U (V * W) ||_F
                # T_a[a, b*p+k] = sum_r U[a, r] V[b, r] W[k, r]
                # Let E_U be (N, M*P) such that E_U[a, b*p+k] = sum_r V[b, r] W[k, r]
                # Then T_a = U @ E_U
                # U.T = lstsq(E_U.T, T_a.T)
                
                # Construct E_U
                # V is (M, R), W is (P, R)
                # V[b, r] * W[k, r]
                # V_expanded: (M, 1, R)
                # W_expanded: (1, P, R)
                # Product: (M, P, R)
                # Sum over R: (M, P)
                # Reshape to (M, P) -> we need (N, M*P) but the dependence on N is only through U in the objective.
                # Wait, T_a[a, ...] depends on a only via U[a, r].
                # The term coupled with U[a, r] is sum_r U[a, r] * (sum_r' V[b, r'] W[k, r'] ... no)
                # T_a[a, idx] = sum_r U[a, r] * (sum_{b,k} delta_{idx, b*p+k} V[b, r] W[k, r])
                # Let G_U be (M*P, R) where G_U[b*p+k, r] = V[b, r] * W[k, r]
                # Then T_a = U @ G_U
                # So U.T = lstsq(G_U.T, T_a.T)
                
                G_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(M * P, R)
                
                try:
                    U_T, _, _, _ = np.linalg.lstsq(G_U.T, T_a.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V: Minimize || T_b - V (U * W) ||_F
                # T_b[b, a*n+k] = sum_r V[b, r] U[a, r] W[k, r]
                # Let G_V be (N*P, R) where G_V[a*n+k, r] = U[a, r] * W[k, r]
                # Then T_b = V @ G_V
                # V.T = lstsq(G_V.T, T_b.T)
                
                G_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N * P, R)
                
                try:
                    V_T, _, _, _ = np.linalg.lstsq(G_V.T, T_b.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check Convergence
                # Compute error: || T_c - E_W_new @ W.T ||
                # Recalculate E_W with new U, V
                E_W_new = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
                T_calc_c = E_W_new @ W.T
                error_c = np.linalg.norm(T_calc_c - T_c)
                
                if error_c < 1e-6:
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    else:
                        # Try to snap or do a few more iterations
                        # Sometimes it converges to a non-integer solution.
                        # We can try a "polish" by running a few more iterations with tighter tol?
                        # Or just accept if error is very small and hope rounding works?
                        # The check_solution already rounds. If it fails, the solution is not half-integer.
                        # We might need to try a different initialization.
                        pass

            if converged:
                break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
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
