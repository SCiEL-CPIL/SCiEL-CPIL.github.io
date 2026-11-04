import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    # Attempt to find a rank 17 decomposition for 2x3x3
    # If found, use it to construct a rank 51 decomposition for 2x3x9
    U_233, V_233, W_233 = search_rank_233(17, deadline)
    
    if U_233 is not None:
        return construct_239_from_233(U_233, V_233, W_233)

    # Fallback: Schoolbook 54
    return construct_schoolbook(n, m, p)

def construct_239_from_233(U_233, V_233, W_233):
    n, m, p = 2, 3, 9
    rank_233 = U_233.shape[1]
    total_rank = 3 * rank_233
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # U is repeated
    for block in range(3):
        U_full[:, block * rank_233 : (block + 1) * rank_233] = U_233
        
    # V mapping: row j*9+k in full -> row j*3+k_local in 233, block k//3
    for j in range(3):
        for k in range(9):
            block = k // 3
            k_local = k % 3
            v_row_full = j * 9 + k
            v_row_233 = j * 3 + k_local
            col_start = block * rank_233
            V_full[v_row_full, col_start:col_start+rank_233] = V_233[v_row_233, :]
            
    # W mapping: row k*2+i in full -> row k_local*2+i in 233, block k//3
    for k in range(9):
        for i in range(2):
            block = k // 3
            k_local = k % 3
            w_row_full = k * 2 + i
            w_row_233 = k_local * 2 + i
            col_start = block * rank_233
            W_full[w_row_full, col_start:col_start+rank_233] = W_233[w_row_233, :]
            
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

def search_rank_233(target_rank, deadline):
    n, m, p = 2, 3, 3
    num_u = n * m # 6
    num_v = m * p # 9
    num_w = p * n # 6
    
    # Precompute target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def calc_error(U, V, W):
        # Efficiently calculate sum of squared errors
        # Res = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # We compute Res - T and sum squares
        Res = np.zeros((num_u, num_v, num_w))
        for r in range(target_rank):
            Res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r], optimize=True)
        diff = Res - T
        return np.sum(diff**2)

    random.seed(42)
    np.random.seed(42)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Simulated Annealing parameters
    # We will run this for the remaining time
    current_time = time.time()
    total_time = 100.0
    start_time = time.time()
    
    # Initialize with random integers
    U = np.random.randint(-1, 2, size=(num_u, target_rank)).astype(float)
    V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
    W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)
    
    current_err = calc_error(U, V, W)
    
    if current_err < best_err:
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
    # SA Parameters
    temp = 10.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Possible moves: change one element by delta
    # Deltas: +/- 0.5, +/- 1.0, +/- 1.5, +/- 2.0
    deltas = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0]
    
    while time.time() < deadline - 5:
        # Pick a random matrix, row, col
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            mat = U
            rows = num_u
        elif mat_idx == 1:
            mat = V
            rows = num_v
        else:
            mat = W
            rows = num_w
            
        row = random.randint(0, rows - 1)
        col = random.randint(0, target_rank - 1)
        
        old_val = mat[row, col]
        delta = random.choice(deltas)
        new_val = old_val + delta
        
        # Optional: Clamp to reasonable range to avoid exploding values
        if new_val > 10: new_val = 10
        if new_val < -10: new_val = -10
        
        mat[row, col] = new_val
        new_err = calc_error(U, V, W)
        
        delta_err = new_err - current_err
        
        if delta_err < 0 or random.random() < np.exp(-delta_err / temp):
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-9:
                    break
            # Cool down if we found a new best? Or just cool down always?
            temp = max(min_temp, temp * cooling_rate)
        else:
            mat[row, col] = old_val
            # Cool down even on rejection? Standard SA cools every step.
            temp = max(min_temp, temp * cooling_rate)
            
        # If we get stuck (no improvement for a long time), restart with new random init
        # Simple heuristic: if temp is low and we haven't improved in 1000 steps, restart
        # For simplicity, we just let it run. The random restarts at the beginning were removed, 
        # but SA should explore.
        
        # Additional check: if best_err is very small, try to snap to half-integers
        if best_err < 1e-6:
            # Snap best solution to nearest half-integer
            bU = np.round(best_U * 2) / 2
            bV = np.round(best_V * 2) / 2
            bW = np.round(best_W * 2) / 2
            snap_err = calc_error(bU, bV, bW)
            if snap_err < 1e-9:
                return bU, bV, bW
            # If snap is better, update best
            if snap_err < best_err:
                best_err = snap_err
                best_U, best_V, best_W = bU, bV, bW
                if best_err < 1e-9:
                    return bU, bV, bW

    if best_err < 1e-9:
        return best_U, best_V, best_W
        
    return None, None, None

def construct_schoolbook(n, m, p):
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
