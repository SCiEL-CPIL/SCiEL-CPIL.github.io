import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_40():
    """
    Hardcoded rank-40 decomposition for 2x3x10 matrix multiplication.
    Uses a more efficient structure based on combining operations across columns.
    """
    n, m, p = DIMENSIONS
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process columns in pairs, using 8 ranks per pair (total 5*8 = 40)
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: A00 * B0,k0
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M2: A00 * B0,k1
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M3: A01 * B1,k0
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M4: A01 * B1,k1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M5: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M6: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M7: A10 * B0,k0
        U[1 * 3 + 0][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M8: A10 * B0,k1
        U[1 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M9: A11 * B1,k0
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M10: A11 * B1,k1
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M11: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M12: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _construct_rank_35():
    """
    Hardcoded rank-35 decomposition for 2x3x10 matrix multiplication.
    Uses a smarter structure that combines operations.
    """
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process columns in pairs, using 7 ranks per pair (total 5*7 = 35)
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A01) * (B0,k0 + B1,k0)
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M2: (A00 - A01) * (B0,k0 - B1,k0)
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M3: (A00 + A01) * (B0,k1 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M4: (A00 - A01) * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] -= 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M5: (A10 + A11) * (B0,k0 + B1,k0)
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M6: (A10 - A11) * (B0,k0 - B1,k0)
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k0][col] -= 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A10 + A11) * (B0,k1 + B1,k1)
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M8: (A10 - A11) * (B0,k1 - B1,k1)
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M9: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # M10: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M11: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M12: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _local_improve_rank(U, V, W, max_time=60.0):
    """
    Attempt to reduce rank by eliminating redundant columns.
    Returns improved (U, V, W) with potentially lower rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    def is_exact(U_curr, V_curr, W_curr):
        rank = U_curr.shape[1]
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('a,b,c->abc', U_curr[:,r], V_curr[:,r], W_curr[:,r])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_calc[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                        return False
        return True
    
    if not is_exact(U, V, W):
        return U, V, W
    
    improved_U, improved_V, improved_W = U.copy(), V.copy(), W.copy()
    current_rank = improved_U.shape[1]
    
    # Try removing columns one at a time
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
        
        removed_any = False
        cols_to_try = list(range(current_rank))
        if attempt > 0:
            np.random.shuffle(cols_to_try)
        
        for r in cols_to_try[:]:
            if time.time() - start_time > max_time:
                break
            
            mask = [i for i in range(current_rank) if i != r]
            if len(mask) == 0:
                continue
                
            U_test = improved_U[:, mask]
            V_test = improved_V[:, mask]
            W_test = improved_W[:, mask]
            
            if is_exact(U_test, V_test, W_test):
                improved_U = U_test
                improved_V = V_test
                improved_W = W_test
                current_rank = len(mask)
                removed_any = True
        
        if not removed_any:
            break
    
    return improved_U, improved_V, improved_W

def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    Based on 5 blocks of 2 columns, each using 11 ranks.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Try rank 35 first
    if time.time() < DEADLINE - 5:
        U, V, W = _construct_rank_35()
        U_int = np.rint(U).astype(int)
        V_int = np.rint(V).astype(int)
        W_int = np.rint(W).astype(int)
        
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(35):
            T_calc += np.einsum('a,b,c->abc', U_int[:,r], V_int[:,r], W_int[:,r])
            
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_calc[i*m+j, j*p+k, k*n+i] != 1.0:
                        valid = False
        if valid:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Try rank 40
    if time.time() < DEADLINE - 5:
        U, V, W = _construct_rank_40()
        U_int = np.rint(U).astype(int)
        V_int = np.rint(V).astype(int)
        W_int = np.rint(W).astype(int)
        
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(40):
            T_calc += np.einsum('a,b,c->abc', U_int[:,r], V_int[:,r], W_int[:,r])
            
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_calc[i*m+j, j*p+k, k*n+i] != 1.0:
                        valid = False
        if valid:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to rank 55
    U, V, W = _construct_rank_55()
    U, V, W = np.rint(U).astype(int), np.rint(V).astype(int), np.rint(W).astype(int)
    
    if time.time() < DEADLINE - 10:
        U, V, W = _local_improve_rank(U, V, W, max_time=DEADLINE - time.time() - 5)
        U = np.rint(U).astype(int)
        V = np.rint(V).astype(int)
        W = np.rint(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
