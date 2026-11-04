import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

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

def _local_search(n, m, p, target_rank, time_budget, start_x=None):
    """
    Attempt to find a decomposition of rank target_rank using local search.
    If start_x is provided, use it as the initial guess.
    """
    start_time = time.time()
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target tensor T[i,j,k,r] = delta(i,r) * delta(j,k)
    
    # Objective function
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

    # Gradient calculation
    def gradient(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            orig = x[i]
            x[i] = orig + eps
            f1 = objective(x)
            x[i] = orig
            g[i] = (f1 - f0) / eps
        return g

    # Try to find a solution
    if start_x is not None:
        x0 = start_x
    else:
        x0 = np.random.randn(size_U * target_rank + size_V * target_rank + size_W * target_rank)
    
    # Use L-BFGS-B with analytical or numerical gradient
    # For speed, we use numerical gradient but limit iterations
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 10000})
        
        if res.fun < 1e-8:
            U_flat = res.x[:size_U * target_rank]
            V_flat = res.x[size_U * target_rank : size_U * target_rank + size_V * target_rank]
            W_flat = res.x[size_U * target_rank + size_V * target_rank :]
            
            U = U_flat.reshape(size_U, target_rank).tolist()
            V = V_flat.reshape(size_V, target_rank).tolist()
            W = W_flat.reshape(size_W, target_rank).tolist()
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, target_rank
    except Exception:
        pass

    return None

def _random_restart_search(n, m, p, target_rank, time_budget):
    """
    Perform random restarts to find a decomposition.
    """
    start_time = time.time()
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_size = size_U * target_rank + size_V * target_rank + size_W * target_rank
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        x0 = np.random.randn(total_size)
        
        result = _local_search(n, m, p, target_rank, 5.0, start_x=x0)
        if result:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Get the baseline decomposition
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
    
    # Strategy: Try to reduce rank from best_rank down to 90
    # We use random restarts and local search
    
    if time_left > 10:
        # Try ranks from best_rank - 1 down to 90
        # Allocate time based on remaining budget
        ranks_to_try = range(best_rank - 1, 89, -1)
        
        for r in ranks_to_try:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Allocate time for this rank
            # If we have a lot of time, use more restarts
            t_budget = min(20, (TIME_LIMIT - 5 - (time.time() - START_TIME)) / 2)
            
            if t_budget < 2:
                break
                
            result = _random_restart_search(n, m, p, r, t_budget)
            if result:
                best_U, best_V, best_W, best_rank = result
                # Continue to try lower ranks if time permits
                # But usually finding one is a big jump.
                # Let's continue the loop to try r-1, etc.
                
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
