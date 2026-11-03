import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target Tensor T
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = start_time + 110
    
    # Known minimal rank is 22
    R = 22
    
    def solve_W(U, V, T, nm, mp, np_, R):
        W = np.zeros((np_, R))
        for c in range(np_):
            M_c = np.zeros((nm * mp, R))
            for r in range(R):
                M_c[:, r] = np.kron(U[:, r], V[:, r])
            rhs = T[:, :, c].flatten()
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M_c, rhs, rcond=None)
                W[c, :] = sol
            except:
                pass
        return W

    def compute_residual(U, V, W, T):
        T_rec = np.einsum('ar, br, cr -> abc', U, V, W)
        return np.linalg.norm(T_rec - T)

    def snap_to_half_int(matrix):
        # Snap values to nearest 0.5
        return np.round(matrix * 2) / 2

    best_U = None
    best_V = None
    best_W = None
    best_res = np.inf
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        
        W = solve_W(U, V, T, nm, mp, np_, R)
        
        res = compute_residual(U, V, W, T)
        
        if res < best_res:
            best_res = res
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if res < 1e-8:
            # Try to snap
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            res_snap = compute_residual(U_snap, V_snap, W_snap, T)
            if res_snap < 1e-6:
                return {"U": U_snap, "V": V_snap, "W": W_snap}
            else:
                # Return float solution if snapping fails but solution is valid
                return {"U": U, "V": V, "W": W}
                
    # If no good solution found, fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
