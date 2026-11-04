import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
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
        
        # Also check if integers work (in case rounding to 0.5 was too aggressive or scaling was off)
        # But np.round(U*2)/2 handles integers correctly (1.0 -> 1.0)
        # What if the solution is scaled by 2? e.g. entries are 0, 1, 2.
        # Then U*2 is 0, 2, 4. Round/2 is 0, 1, 2. Correct.
        # What if entries are 0, 0.5, 1?
        # U*2 is 0, 1, 2. Round/2 is 0, 0.5, 1. Correct.
        
        return False, None, None, None

    # Pre-compute matricizations for speed
    T_c = T.reshape(N * M, P)
    T_a_vec = T.reshape(N, M * P)
    T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)

    # Search ranks from 35 down to 25
    # Higher ranks are more likely to have solutions
    # We allocate more attempts to higher ranks initially
    ranks_to_try = list(range(35, 24, -1))
    
    for R in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Number of attempts: more for higher ranks
        attempts = 10 if R > 30 else 5
        
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(1000 + R * 100 + attempt)
            # Scale initialization to help convergence?
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            converged = False
            # Max iterations
            max_iters = 50
            
            for it in range(max_iters):
                if time.time() - start_time > timeout:
                    break
                
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                # Calculate error efficiently?
                # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                # error = np.linalg.norm(T_calc - T)
                
                # Faster error check using matricizations
                # ||T - U circ V circ W||_F^2 = ||T_c - E_W W^T||_F^2 ? No.
                # T_c[i, c] = sum_r U[a,r] V[b,r] W[c,r]
                # E_W[i, r] = U[a,r] V[b,r]
                # So T_c should be E_W @ W.T
                # Error^2 = ||T_c - E_W @ W.T||_F^2
                # We already computed E_W.
                # Let's compute the error from the last update step.
                # Actually, we can just compute the norm difference.
                
                T_calc_c = E_W @ W.T
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
                        # If not valid, we might be in a local min or the solution isn't half-integer.
                        # We can try a few more iterations to see if it snaps.
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
