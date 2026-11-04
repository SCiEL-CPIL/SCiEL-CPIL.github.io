import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # m1 = (a+d)(e+h)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0  # C00

    # m2 = (c+d)(e+f)
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0; V[1][1] = 1.0
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11

    # m3 = a(f-h)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11

    # m4 = d(g-e)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0  # C00
    W[2][3] = 1.0  # C10

    # m5 = (a+b)h
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0 # C00
    W[1][4] = 1.0  # C01

    # m6 = (c-d)e
    U[2][5] = 1.0; U[3][5] = -1.0
    V[0][5] = 1.0
    W[3][5] = 1.0  # C11

    # m7 = (b-d)(f+g)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[2][6] = 1.0
    W[0][6] = 1.0  # C00

    return U, V, W, rank

def _decompose_2x3x3():
    """
    Decompose 2x3 * 3x3 -> 2x3.
    Rank 17.
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. 2x2 * 2x2 -> 2x2 (Strassen)
    U1, V1, W1, r1 = _strassen_2x2()
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i * 3 + j][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j * 3 + k][col + c] = V1[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k * 2 + i][col + c] = W1[k * 2 + i][c]
    col += r1
    
    # 2. 2x1 * 1x2 -> 2x2 (Schoolbook)
    U2, V2, W2, r2 = _schoolbook(2, 1, 2)
    for i in range(2):
        for c in range(r2):
            U[i * 3 + 2][col + c] = U2[i * 1][c]
    for j in range(1):
        for k in range(2):
            for c in range(r2):
                V[(2 + j) * 3 + k][col + c] = V2[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k * 2 + i][col + c] += W2[k * 2 + i][c]
    col += r2
    
    # 3. 2x2 * 2x1 -> 2x1 (Schoolbook)
    U3, V3, W3, r3 = _schoolbook(2, 2, 1)
    for i in range(2):
        for j in range(2):
            for c in range(r3):
                U[i * 3 + j][col + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(1):
            for c in range(r3):
                V[j * 3 + (2 + k)][col + c] = V3[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r3):
                W[(2 + k) * 2 + i][col + c] = W3[k * 2 + i][c]
    col += r3
    
    # 4. 2x1 * 1x1 -> 2x1 (Schoolbook)
    U4, V4, W4, r4 = _schoolbook(2, 1, 1)
    for i in range(2):
        for c in range(r4):
            U[i * 3 + 2][col + c] = U4[i * 1][c]
    for j in range(1):
        for k in range(1):
            for c in range(r4):
                V[(2 + j) * 3 + (2 + k)][col + c] = V4[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r4):
                W[(2 + k) * 2 + i][col + c] += W4[k * 2 + i][c]
    col += r4
    
    return U, V, W, col

def _decompose_4x5x5():
    """
    Decompose 4x5 * 5x5 -> 4x5.
    Rank 96.
    """
    n, m, p = 4, 5, 5
    total_rank = 96
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for block in range(2):
        i_offset = block * 2
        
        # Sub 1: 2x2 * 2x2 -> 2x2 (Strassen)
        U1, V1, W1, r1 = _strassen_2x2()
        for i in range(2):
            for j in range(2):
                for c in range(r1):
                    U[(i_offset + i) * m + j][col + c] = U1[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V[j * p + k][col + c] = V1[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r1):
                    W[k * n + (i_offset + i)][col + c] = W1[k * 2 + i][c]
        col += r1
        
        # Sub 2: 2x3 * 3x2 -> 2x2 (Schoolbook)
        U2, V2, W2, r2 = _schoolbook(2, 3, 2)
        for i in range(2):
            for j in range(3):
                for c in range(r2):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U2[i * 3 + j][c]
        for j in range(3):
            for k in range(2):
                for c in range(r2):
                    V[(2 + j) * p + k][col + c] = V2[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r2):
                    W[k * n + (i_offset + i)][col + c] += W2[k * 2 + i][c]
        col += r2
        
        # Sub 3: 2x2 * 2x3 -> 2x3 (Schoolbook)
        U3, V3, W3, r3 = _schoolbook(2, 2, 3)
        for i in range(2):
            for j in range(2):
                for c in range(r3):
                    U[(i_offset + i) * m + j][col + c] = U3[i * 2 + j][c]
        for j in range(2):
            for k in range(3):
                for c in range(r3):
                    V[j * p + (2 + k)][col + c] = V3[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r3):
                    W[(2 + k) * n + (i_offset + i)][col + c] = W3[k * 2 + i][c]
        col += r3
        
        # Sub 4: 2x3 * 3x3 -> 2x3 (Improved, rank 17)
        U4, V4, W4, r4 = _decompose_2x3x3()
        for i in range(2):
            for j in range(3):
                for c in range(r4):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U4[i * 3 + j][c]
        for j in range(3):
            for k in range(3):
                for c in range(r4):
                    V[(2 + j) * p + (2 + k)][col + c] = V4[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r4):
                    W[(2 + k) * n + (i_offset + i)][col + c] += W4[k * 2 + i][c]
        col += r4
    
    return U, V, W, col

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+r, c]
    # We want T[i,j,k,r] = delta_{i,r} delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _local_search(n, m, p, target_rank, time_budget, init_x=None):
    """
    Attempt to find a decomposition of rank target_rank using local search.
    """
    start_time = time.time()
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    best_result = None
    best_error = float('inf')
    
    # If no initialization provided, use random restarts
    if init_x is None:
        num_restarts = 5
        if time_budget < 10:
            num_restarts = 2
        elif time_budget < 30:
            num_restarts = 3
            
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            x0 = np.random.randn(size_U * target_rank + size_V * target_rank + size_W * target_rank)
            res, err = _run_minimize(n, m, p, target_rank, x0, time_budget - (time.time() - start_time))
            if err < best_error:
                best_error = err
                best_result = res
            if best_error < 1e-9:
                break
    else:
        # Use provided initialization (e.g., from a higher rank solution with a column zeroed out)
        res, err = _run_minimize(n, m, p, target_rank, init_x, time_budget)
        best_error = err
        best_result = res

    if best_result is not None and best_error < 1e-9:
        U_flat = best_result[:size_U * target_rank]
        V_flat = best_result[size_U * target_rank : size_U * target_rank + size_V * target_rank]
        W_flat = best_result[size_U * target_rank + size_V * target_rank :]
        
        U = U_flat.reshape(size_U, target_rank).tolist()
        V = V_flat.reshape(size_V, target_rank).tolist()
        W = W_flat.reshape(size_W, target_rank).tolist()
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W, target_rank

    return None

def _run_minimize(n, m, p, target_rank, x0, time_budget):
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    def objective(x):
        U_flat = x[:size_U * target_rank]
        V_flat = x[size_U * target_rank : size_U * target_rank + size_V * target_rank]
        W_flat = x[size_U * target_rank + size_V * target_rank :]
        
        U = U_flat.reshape(size_U, target_rank)
        V = V_flat.reshape(size_V, target_rank)
        W = W_flat.reshape(size_W, target_rank)
        
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        T_calc = np.einsum('ijc,jkc,krc->ijkr', U_3d, V_3d, W_3d)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                val_diag = T_calc[i, j, j, i]
                error += (val_diag - 1.0) ** 2
                ss = np.sum(T_calc[i, j, :, :] ** 2)
                error += ss - 2.0 * val_diag + 1.0
        return error

    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9})
        return res.x, res.fun
    except Exception:
        return x0, float('inf')

def construct():
    n, m, p = DIMENSIONS
    
    # Get the baseline decomposition (Rank 96)
    try:
        U, V, W, rank = _decompose_4x5x5()
        if _verify_decomposition(U, V, W, n, m, p):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            U, V, W, best_rank = _schoolbook(n, m, p)
            best_U, best_V, best_W = U, V, W
    except Exception:
        U, V, W, best_rank = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W

    # Try to improve with local search
    current_time = time.time()
    elapsed = current_time - START_TIME
    time_left = TIME_LIMIT - elapsed
    
    if time_left > 10:
        # Try rank 95 first
        # Initialize by taking the rank 96 solution and zeroing out the last column of U, V, W
        # This is a heuristic initialization for the lower rank problem
        init_x_95 = np.zeros((n*m + m*p + p*n) * 95)
        
        # Copy first 95 columns
        for c in range(95):
            for i in range(n*m):
                init_x_95[i*95 + c] = U[i][c]
            for j in range(m*p):
                init_x_95[(n*m + j)*95 + c] = V[j][c]
            for k in range(p*n):
                init_x_95[(n*m + m*p + k)*95 + c] = W[k][c]
        
        # We need to adjust the dimensions for the optimizer
        # The optimizer expects a flat array of length (n*m + m*p + p*n) * target_rank
        # My init_x_95 construction above is slightly off in indexing. Let's do it properly.
        
        U_flat_95 = np.array(U)[:, :95].flatten()
        V_flat_95 = np.array(V)[:, :95].flatten()
        W_flat_95 = np.array(W)[:, :95].flatten()
        init_x_95 = np.concatenate([U_flat_95, V_flat_95, W_flat_95])
        
        result = _local_search(n, m, p, 95, min(30, time_left - 5), init_x_95)
        if result:
            best_U, best_V, best_W, best_rank = result
            time_left = TIME_LIMIT - (time.time() - START_TIME)
        
        # If 95 failed, try 94 with a new random start or modified init
        if time_left > 10 and best_rank > 94:
            # Use random restarts for 94
            result = _local_search(n, m, p, 94, min(20, time_left - 5))
            if result:
                best_U, best_V, best_W, best_rank = result

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
