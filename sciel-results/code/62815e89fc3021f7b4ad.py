import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = None
    best_rank = n * m * p

    def verify_solution(U, V, W, R):
        """Verify that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k"""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def solve_for_W(U, V, R):
        """Given U and V, solve linear system for W."""
        W = np.zeros((p * n, R))
        for i in range(n):
            for k in range(p):
                # B[j, r] = U[i*m+j, r] * V[j*p+k, r]
                B = np.zeros((m, R))
                for j in range(m):
                    B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve B w = ones(m)
                try:
                    w, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    if not np.allclose(B @ w, np.ones(m), atol=1e-8):
                        return None
                    W[k * n + i, :] = w
                except:
                    return None
        return W

    def snap_and_verify(U, V, W, R):
        """Snap to half-integers and verify exactness."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify_solution(U_s, V_s, W_s, R):
            return U_s, V_s, W_s
        # Try integer snap
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_solution(U_i, V_i, W_i, R):
            return U_i, V_i, W_i
        return None

    def polish(U, V, W, R):
        """High-precision polish using least squares on all variables."""
        from scipy.optimize import least_squares
        
        def compute_error(params):
            U_p = params[:n*m*R].reshape(n, m, R)
            V_p = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
            W_p = params[n*m*R + m*p*R:].reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijk', U_p, V_p, W_p)
            return (T - 1.0).flatten()
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = least_squares(compute_error, params, method='lm', max_nfev=200)
            if res.cost < 1e-12:
                U_sol = res.x[:n*m*R].reshape(n*m, R)
                V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                snapped = snap_and_verify(U_sol, V_sol, W_sol, R)
                if snapped is not None:
                    return snapped
        except:
            pass
        return None

    # Strategy: For each rank R, try random U,V, solve for W, verify
    # Use structured random initialization for better chances
    
    for R in range(20, best_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        num_attempts = 200 if R < 40 else 100
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Multiple initialization strategies
            if attempt % 4 == 0:
                # Pure random
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
            elif attempt % 4 == 1:
                # Sparse random (many zeros)
                U = np.random.choice([-1, 0, 1], size=(n * m, R))
                V = np.random.choice([-1, 0, 1], size=(m * p, R))
            elif attempt % 4 == 2:
                # Binary random
                U = np.random.choice([0, 1], size=(n * m, R)).astype(float)
                V = np.random.choice([0, 1], size=(m * p, R)).astype(float)
            else:
                # Small integers
                U = np.random.randint(-2, 3, size=(n * m, R)).astype(float)
                V = np.random.randint(-2, 3, size=(m * p, R)).astype(float)
            
            W = solve_for_W(U, V, R)
            if W is None:
                continue
            
            # Snap and verify
            snapped = snap_and_verify(U, V, W, R)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                best_rank = R
                best_result = {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }
                found = True
                break
            else:
                # Try polish
                polished = polish(U, V, W, R)
                if polished is not None:
                    U_s, V_s, W_s = polished
                    best_rank = R
                    best_result = {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
