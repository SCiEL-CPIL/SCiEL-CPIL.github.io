import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook fallback
    def schoolbook():
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

    fallback = schoolbook()
    
    def verify_and_snap(U, V, W, tol=1e-9):
        # Check if U, V, W are close to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check exactness with snapped values
        # res[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            U_c = U_snap[:, c].reshape(n, m)
            V_c = V_snap[:, c].reshape(m, p)
            W_c = W_snap[:, c].reshape(p, n)
            # res[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
            U_part = U_c[:, :, np.newaxis]
            V_part = V_c[np.newaxis, :, :]
            W_part = W_c.T[:, np.newaxis, :] # W_c.T is (n, p), so W_part[i, 0, k] = W_c[k, i]
            res += U_part * V_part * W_part
            
        if np.allclose(res, 1.0, atol=tol):
            return U_snap, V_snap, W_snap
        return None

    def solve_U(U, V, W):
        # Solve for U given V, W
        # For each i, j, solve A_ij x = 1
        # A_ij is (p, r), A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, U.shape[1]))
                for k in range(p):
                    for c in range(U.shape[1]):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                b = np.ones(p)
                # Solve A x = b using least squares
                try:
                    U_new[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        return U_new

    def solve_V(U, V, W):
        # Solve for V given U, W
        # For each j, k, solve A_jk x = 1
        # A_jk is (m, r)? No.
        # Eq: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
        # Variable is V[j*p+k, c].
        # For fixed j, k, the variable is vector v_jk of length r.
        # It appears in equations for all i.
        # Eq for (i, j, k): sum_c v_jk[c] * (U[i*m+j, c] * W[k*n+i, c]) = 1
        # So A_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, V.shape[1]))
                for i in range(n):
                    for c in range(V.shape[1]):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                b = np.ones(n)
                try:
                    V_new[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        return V_new

    def solve_W(U, V, W):
        # Solve for W given U, V
        # For each k, i, solve A_ki x = 1
        # Variable is W[k*n+i, c].
        # Appears in equations for all j.
        # Eq for (i, j, k): sum_c w_ki[c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
        # A_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, W.shape[1]))
                for j in range(m):
                    for c in range(W.shape[1]):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                try:
                    W_new[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        return W_new

    # Try ranks from 30 down to 20
    for r in range(30, 19, -1):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time remaining
        time_left = deadline - time.time()
        max_restarts = max(1, int(time_left / 2.0)) # ~2s per restart estimate
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            if r == 30:
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
            else:
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            
            # ALS Iterations
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                # Check convergence/error
                # Calculate error
                res = np.zeros((n, m, p))
                for c in range(r):
                    U_c = U[:, c].reshape(n, m)
                    V_c = V[:, c].reshape(m, p)
                    W_c = W[:, c].reshape(p, n)
                    res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
                error = np.sum((res - 1.0)**2)
                
                if error < 1e-12:
                    converged = True
                    break
                    
            if time.time() > deadline:
                break
                
            if converged:
                result = verify_and_snap(U, V, W)
                if result:
                    U_sol, V_sol, W_sol = result
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

    return fallback

# EVOLVE-BLOCK-END
