import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def solve_for_U(V, W):
        # A_U shape: (m*p*p*n, r)
        r = V.shape[1]
        A_U = np.zeros((m*p*p*n, r))
        for c in range(r):
            A_U[:, c] = np.kron(V[:, c], W[:, c])
        
        U = np.zeros((n*m, r))
        for a in range(n*m):
            t_a = T[a, :, :].flatten()
            # Solve A_U U_a^T = t_a
            # U_a is 1 x r
            U[a, :] = np.linalg.lstsq(A_U, t_a, rcond=None)[0]
        return U

    def solve_for_V(U, W):
        r = U.shape[1]
        A_V = np.zeros((n*m*p*n, r))
        for c in range(r):
            A_V[:, c] = np.kron(U[:, c], W[:, c])
        
        V = np.zeros((m*p, r))
        for b in range(m*p):
            t_b = T[:, b, :].flatten()
            V[b, :] = np.linalg.lstsq(A_V, t_b, rcond=None)[0]
        return V

    def solve_for_W(U, V):
        r = U.shape[1]
        A_W = np.zeros((n*m*m*p, r))
        for c in range(r):
            A_W[:, c] = np.kron(U[:, c], V[:, c])
        
        W = np.zeros((p*n, r))
        for d in range(p*n):
            t_d = T[:, :, d].flatten()
            W[d, :] = np.linalg.lstsq(A_W, t_d, rcond=None)[0]
        return W

    def error(U, V, W):
        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.sum((T_calc - T)**2)

    best_result = None
    start_time = time.time()
    deadline = start_time + 140
    
    for r in range(19, 27):
        if time.time() > deadline:
            break
            
        found = False
        max_starts = 20
        for start in range(max_starts):
            if time.time() > deadline:
                break
                
            # Random init
            np.random.seed(start)
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # ALS iterations
            for it in range(100):
                U = solve_for_U(V, W)
                V_new = solve_for_V(U, W)
                W_new = solve_for_W(U, V_new)
                
                if np.allclose(V, V_new, atol=1e-6) and np.allclose(W, W_new, atol=1e-6):
                    break
                
                V = V_new
                W = W_new
                
                # Check error
                err = error(U, V, W)
                if err < 1e-6:
                    # Snap and verify
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    T_calc = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
                    if np.allclose(T_calc, T, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    found = True # Found low error but not exact half-integer?
            
            if found:
                # If we found a low error solution, maybe try more iterations or different snap?
                # For now, continue to next start or rank
                pass

    # Fallback
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
