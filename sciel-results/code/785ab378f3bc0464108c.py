import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    return np.allclose(T_decomp, T, atol=tol)

def _solve_subproblem(n, m, p, target_rank, max_time=10.0):
    """
    Try to find a decomposition of rank target_rank for n x m x p multiplication.
    Uses simulated annealing-like local search.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    values = [-1, -0.5, 0, 0.5, 1]
    
    def compute_error(U, V, W):
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.linalg.norm(T_decomp - T)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Use multiple restarts
    while time.time() < deadline:
        U = np.random.choice(values, size=(n*m, target_rank))
        V = np.random.choice(values, size=(m*p, target_rank))
        W = np.random.choice(values, size=(p*n, target_rank))
        
        err = compute_error(U, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            return U, V, W
            
        # Local search with temperature
        temp = 0.5
        for it in range(2000):
            if time.time() > deadline:
                break
                
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                arr = U
                rows, cols = n*m, target_rank
            elif mat_choice == 'V':
                arr = V
                rows, cols = m*p, target_rank
            else:
                arr = W
                rows, cols = p*n, target_rank
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            new_val = random.choice(values)
            
            if new_val == old_val:
                continue
                
            arr[row, col] = new_val
            new_err = compute_error(U, V, W)
            
            delta = new_err - err
            if delta < 0:
                err = new_err
                if err < 1e-8:
                    return U, V, W
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Accept with probability exp(-delta / temp)
                if random.random() < np.exp(-delta / temp):
                    err = new_err
                else:
                    arr[row, col] = old_val
                    
            # Cool down
            temp *= 0.999
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    p1 = 3
    p2 = 2
    
    # Solve 3x3x3
    res_333 = None
    rank_333 = 27
    # Try 23, 24, 25
    for r in [23, 24, 25]:
        if time.time() > deadline - 30:
            break
        res_333 = _solve_subproblem(3, 3, p1, r, max_time=15.0)
        if res_333:
            rank_333 = r
            break
            
    if not res_333:
        # Fall back to schoolbook
        U_333 = [[0]*rank_333 for _ in range(9)]
        V_333 = [[0]*rank_333 for _ in range(9)]
        W_333 = [[0]*rank_333 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p1):
                    U_333[i*3+j][col] = 1
                    V_333[j*p1+k][col] = 1
                    W_333[k*3+i][col] = 1
                    col += 1
        res_333 = (np.array(U_333), np.array(V_333), np.array(W_333))
        rank_333 = 27

    # Solve 3x3x2
    res_332 = None
    rank_332 = 18
    # Try 16, 17
    for r in [16, 17]:
        if time.time() > deadline - 15:
            break
        res_332 = _solve_subproblem(3, 3, p2, r, max_time=10.0)
        if res_332:
            rank_332 = r
            break
            
    if not res_332:
        # Fall back to schoolbook
        U_332 = [[0]*rank_332 for _ in range(9)]
        V_332 = [[0]*rank_332 for _ in range(6)]
        W_332 = [[0]*rank_332 for _ in range(6)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p2):
                    U_332[i*3+j][col] = 1
                    V_332[j*p2+k][col] = 1
                    W_332[k*3+i][col] = 1
                    col += 1
        res_332 = (np.array(U_332), np.array(V_332), np.array(W_332))
        rank_332 = 18

    U_333, V_333, W_333 = res_333
    U_332, V_332, W_332 = res_332
    
    total_rank = rank_333 + rank_332
    
    # Construct combined matrices
    U_total = np.zeros((9, total_rank))
    V_total = np.zeros((15, total_rank))
    W_total = np.zeros((15, total_rank))
    
    # Part 1: 3x3x3 (k=0,1,2)
    U_total[:, :rank_333] = U_333
    for j in range(3):
        for k in range(p1):
            V_total[j*5+k, :rank_333] = V_333[j*3+k, :rank_333]
    W_total[:9, :rank_333] = W_333
    
    # Part 2: 3x3x2 (k=3,4)
    U_total[:, rank_333:] = U_332
    for j in range(3):
        for kp in range(p2):
            k = p1 + kp
            V_total[j*5+k, rank_333:] = V_332[j*2+kp, :]
    for kp in range(p2):
        k = p1 + kp
        W_total[k*3:(k+1)*3, rank_333:] = W_332[kp*3:(kp+1)*3, :]
        
    # Convert to lists of lists
    U_best = [[float(x) for x in row] for row in U_total]
    V_best = [[float(x) for x in row] for row in V_total]
    W_best = [[float(x) for x in row] for row in W_total]
    
    # Final verification
    if not _check_decomposition(U_best, V_best, W_best, n, m, p, total_rank):
        # Fall back to full schoolbook
        schoolbook_rank = n * m * p
        U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j][column] = 1
                    V_fallback[j * p + k][column] = 1
                    W_fallback[k * n + i][column] = 1
                    column += 1
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        
    return {"U": U_best, "V": V_best, "W": W_best}
