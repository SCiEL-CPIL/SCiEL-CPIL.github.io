import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_BUDGET = 135.0

def _tensor_index(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0.0)

def _build_rank22_decomposition(n, m, p):
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        col = col_start
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    c1 = {'00': 0, '01': 2, '10': 1, '11': 3}
    col = _add_strassen_block(a1, b1, c1, 0)

    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    col = _add_strassen_block(a1, b2, c2, col)

    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return U, V, W

def _try_reduce_nonlinear(U, V, W, T, target_rank, max_attempts=50):
    n, m, p = T.shape
    n_u, n_v, n_w = n*m, m*p, p*n
    current_rank = U.shape[1]
    
    # Flatten U, V, W
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
    def unflatten(x):
        u_size = n_u * target_rank
        v_size = n_v * target_rank
        U_r = x[:u_size].reshape(n_u, target_rank)
        V_r = x[u_size:u_size+v_size].reshape(n_v, target_rank)
        W_r = x[u_size+v_size:].reshape(n_w, target_rank)
        return U_r, V_r, W_r

    def residual(x):
        U_r, V_r, W_r = unflatten(x)
        rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return (rec - T).flatten()

    # Try to remove (current_rank - target_rank) terms
    num_remove = current_rank - target_rank
    if num_remove <= 0:
        return U, V, W

    # Generate random subsets of columns to remove (limited attempts for speed)
    import itertools
    indices_to_remove = list(itertools.combinations(range(current_rank), num_remove))
    
    # Limit total attempts to avoid timeout
    max_attempts_actual = min(max_attempts, len(indices_to_remove))
    for i in range(max_attempts_actual):
        if time.time() - START_TIME > TIME_BUDGET - 15:
            break
        
        idxs = indices_to_remove[i]
        
        # Create initial guess with these columns zeroed
        U_init = U.copy()
        V_init = V.copy()
        W_init = W.copy()
        for idx in idxs:
            U_init[:, idx] = 0
            V_init[:, idx] = 0
            W_init[:, idx] = 0
            
        x0 = flatten(U_init, V_init, W_init)
        
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=500)
            if res.cost < 1e-12:
                U_sol, V_sol, W_sol = unflatten(res.x)
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                if _verify(U_round, V_round, W_round, T):
                    return U_round, V_round, W_round
        except Exception:
            continue
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _tensor_index(n, m, p)
    
    U, V, W = _build_rank22_decomposition(n, m, p)
    
    if not _verify(U, V, W, T):
        # Fallback schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_fallback = U.copy()
    V_fallback = V.copy()
    W_fallback = W.copy()

    # Try to reduce to rank 20 (limited attempts for speed)
    if time.time() - START_TIME < TIME_BUDGET - 30:
        U20, V20, W20 = _try_reduce_nonlinear(U, V, W, T, 20, max_attempts=10)
        if _verify(U20, V20, W20, T):
            U, V, W = U20, V20, W20
            current_rank = 20
        else:
            current_rank = 22

    # If rank 20 failed, try rank 21 (limited attempts)
    if current_rank == 22 and time.time() - START_TIME < TIME_BUDGET - 15:
        U21, V21, W21 = _try_reduce_nonlinear(U, V, W, T, 21, max_attempts=10)
        if _verify(U21, V21, W21, T):
            U, V, W = U21, V21, W21

    # Final rounding and verification
    U = np.round(U, 10)
    V = np.round(V, 10)
    W = np.round(W, 10)
    
    if not _verify(U, V, W, T):
        U = np.round(U_fallback, 10)
        V = np.round(V_fallback, 10)
        W = np.round(W_fallback, 10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
    T = _tensor_index(2, 3, 4)
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify(U, V, W, T)}")
