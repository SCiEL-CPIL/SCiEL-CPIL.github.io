import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 110.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_333(U, V, W, tol=1e-10):
    n, m, p = 3, 3, 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if U.shape != (9, rank) or V.shape != (9, rank) or W.shape != (9, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _search_333_rank_23():
    # Search for rank 23 decomposition of 3x3x3
    # Use local search on 9x23 matrices
    n, m, p = 3, 3, 3
    rank = 23
    U = np.random.choice([0, 0.5, 1, 1.5, 2], size=(9, rank))
    V = np.random.choice([0, 0.5, 1, 1.5, 2], size=(9, rank))
    W = np.random.choice([0, 0.5, 1, 1.5, 2], size=(9, rank))
    
    # Vectorized error calculation
    def get_error(U, V, W):
        T = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.sum((T - 1.0)**2)
    
    current_error = get_error(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    limit = 20.0 # 20 seconds for this search
    
    while time.time() - start_time < limit:
        # Random perturbation
        mat_idx = random.randint(0, 2)
        if mat_idx == 0: mat = U
        elif mat_idx == 1: mat = V
        else: mat = W
        
        r = random.randint(0, 8)
        c = random.randint(0, rank - 1)
        
        old_val = mat[r, c]
        new_val = old_val + random.choice([-1.0, -0.5, 0.5, 1.0])
        if new_val < -5: new_val = -5
        if new_val > 5: new_val = 5
        
        mat[r, c] = new_val
        new_error = get_error(U, V, W)
        
        if new_error < current_error:
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error < 1e-10:
                    return best_U, best_V, best_W
        else:
            # Revert if not improving (hill climbing)
            mat[r, c] = old_val
            
    return None

def _construct_46_from_23(U3, V3, W3):
    # Construct 3x3x6 decomposition from 3x3x3 decomposition
    # Rank 46
    U_new = np.hstack([U3, U3])
    
    V_new = np.zeros((18, 46))
    # V rows 0-8 correspond to B1 (cols 0-2 of B)
    V_new[0:9, 0:23] = V3
    # V rows 9-17 correspond to B2 (cols 3-5 of B)
    V_new[9:18, 23:46] = V3
    
    W_new = np.zeros((18, 46))
    # W rows 0-8 correspond to C1 (cols 0-2 of C)
    # Index k*n+i. k=0,1,2.
    W_new[0:9, 0:23] = W3
    # W rows 9-17 correspond to C2 (cols 3-5 of C)
    # Index k*n+i. k=3,4,5.
    # W3 index is k'*3+i.
    # W_new index is (3+k')*3+i = 9 + k'*3 + i.
    W_new[9:18, 23:46] = W3
    
    return {
        "U": U_new.tolist(),
        "V": V_new.tolist(),
        "W": W_new.tolist()
    }

def construct():
    start_time = time.time()
    
    # 1. Try to find rank 23 decomposition for 3x3x3
    # If found, construct rank 46 solution for 3x3x6
    result_23 = _search_333_rank_23()
    if result_23:
        U3, V3, W3 = result_23
        if _verify_333(U3, V3, W3):
            sol = _construct_46_from_23(U3, V3, W3)
            if _verify(sol['U'], sol['V'], sol['W']):
                return sol
    
    # 2. Fallback: Schoolbook
    return _schoolbook()
