import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school
    
    # Known rank-11 solution (Strassen + schoolbook for col 2)
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    U_known[0][0] = 1; U_known[3][0] = 1
    V_known[0][0] = 1; V_known[4][0] = 1
    U_known[2][1] = 1; U_known[3][1] = 1
    V_known[0][1] = 1
    U_known[0][2] = 1
    V_known[1][2] = 1; V_known[4][2] = -1
    U_known[3][3] = 1
    V_known[3][3] = 1; V_known[0][3] = -1
    U_known[0][4] = 1; U_known[1][4] = 1
    V_known[4][4] = 1
    U_known[2][5] = 1; U_known[0][5] = -1
    V_known[0][5] = 1; V_known[1][5] = 1
    U_known[1][6] = 1; U_known[3][6] = -1
    V_known[3][6] = 1; V_known[4][6] = 1
    W_known[0][0] = 1; W_known[0][3] = 1; W_known[0][4] = -1; W_known[0][6] = 1
    W_known[2][2] = 1; W_known[2][4] = 1
    W_known[1][1] = 1; W_known[1][3] = 1
    W_known[3][0] = 1; W_known[3][1] = -1; W_known[3][2] = 1; W_known[3][5] = 1
    U_known[0][7] = 1
    V_known[2][7] = 1
    U_known[1][8] = 1
    V_known[5][8] = 1
    U_known[2][9] = 1
    V_known[2][9] = 1
    U_known[3][10] = 1
    V_known[5][10] = 1
    W_known[4][7] = 1; W_known[4][8] = 1
    W_known[5][9] = 1; W_known[5][10] = 1
    
    # Update best with rank-11 if valid
    U11_arr = np.array(U_known, dtype=float)
    V11_arr = np.array(V_known, dtype=float)
    W11_arr = np.array(W_known, dtype=float)
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        """Compute tensor from factor matrices using einsum for speed"""
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W):
        """Check if decomposition is exact"""
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-9)
    
    if verify(U11_arr, V11_arr, W11_arr):
        best_U, best_V, best_W = U11_arr.tolist(), V11_arr.tolist(), W11_arr.tolist()
        best_rank = rank_known
    
    # Try to find rank-10 decomposition using structured search
    # Strategy: use the rank-11 solution as warm start and try to merge terms
    # or find a valid rank-10 via numerical optimization with allowed values
    
    allowed_vals = [float(x) for x in range(-6, 7)]  # -3, -2.5, -2, ..., 2.5, 3
    
    def try_rank(target_rank, max_attempts=200):
        nonlocal best_U, best_V, best_W, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                return False
            
            # Warm start from rank-11 solution: take first target_rank columns
            if attempt < 50:
                # Try structured initialization from rank-11
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    U[:, r] = U11_arr[:, r]
                    V[:, r] = V11_arr[:, r]
                    W[:, r] = W11_arr[:, r]
                # Add small random perturbation from allowed values
                for r in range(target_rank):
                    U[:, r] += np.random.choice(allowed_vals, size=n*m) * 0.1
                    V[:, r] += np.random.choice(allowed_vals, size=m*p) * 0.1
                    W[:, r] += np.random.choice(allowed_vals, size=p*n) * 0.1
            else:
                # Random initialization with allowed values
                U = np.random.choice(allowed_vals, size=(n*m, target_rank))
                V = np.random.choice(allowed_vals, size=(m*p, target_rank))
                W = np.random.choice(allowed_vals, size=(p*n, target_rank))
            
            if verify(U, V, W):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                return True
        
        return False
    
    # Try rank 10
    if try_rank(10, max_attempts=300):
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
