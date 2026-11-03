# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def _find_rank_r_3x3x3(rank, deadline, num_restarts=20):
    """
    Attempt to find a rank-r decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (9, rank) if found, else None.
    """
    n, m, p = 3, 3, 3
    
    def residuals(x):
        size = 9 * rank
        U = x[0:size].reshape(9, rank)
        V = x[size:2*size].reshape(9, rank)
        W = x[2*size:3*size].reshape(9, rank)
        
        res = np.zeros(27)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    best_err = float('inf')
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(attempt + rank)
        x0 = np.random.choice([-1, 0, 1], size=3 * 9 * rank)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=500)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_solution = sol.x
                
                if err < 1e-8:
                    break
        except Exception:
            continue

    if best_solution is None:
        return None

    size = 9 * rank
    U = best_solution[0:size].reshape(9, rank)
    V = best_solution[size:2*size].reshape(9, rank)
    W = best_solution[2*size:3*size].reshape(9, rank)
    
    U_int = np.round(U * 2) / 2
    V_int = np.round(V * 2) / 2
    W_int = np.round(W * 2) / 2
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Try ranks 23, 24, 25, 26
    for rank3 in [23, 24, 25, 26]:
        result = _find_rank_r_3x3x3(rank3, deadline)
        if result is not None:
            U3, V3, W3 = result
            total_rank = 2 * rank3
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Block 1
            for c in range(rank3):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][c] = U3[i * 3 + j][c]
                for j in range(m):
                    for k in range(3):
                        V[j * p + k][c] = V3[j * 3 + k][c]
                for k in range(3):
                    for i in range(n):
                        W[k * n + i][c] = W3[k * 3 + i][c]
            
            # Block 2
            for c in range(rank3):
                offset = rank3 + c
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][offset] = U3[i * 3 + j][c]
                for j in range(m):
                    for k in range(3):
                        V[j * p + (k + 3)][offset] = V3[j * 3 + k][c]
                for k in range(3):
                    for i in range(n):
                        W[(k + 3) * n + i][offset] = W3[k * 3 + i][c]
            
            # Verify
            def verify_decomposition(U_mat, V_mat, W_mat, rank, n, m, p):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                            if abs(s - 1.0) > 1e-9:
                                return False
                return True
            
            if verify_decomposition(U, V, W, total_rank, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
