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
    
    # Precompute target tensor T (flattened or indexed)
    # T[a, b, c] = 1 if valid, 0 else
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def calc_error(U, V, W):
        # Efficiently calculate sum of squared errors
        # T - sum_r U[:,r] outer V[:,r] outer W[:,r]
        # We can compute this by iterating r and adding to a tensor
        # Or using einsum
        # Result = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow. 
        # Let's try:
        # Res = np.zeros((num_u, num_v, num_w))
        # for r in range(target_rank):
        #    Res += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # return np.sum((Res - T)**2)
        
        # Faster:
        # Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # return np.sum((Res - T)**2)
        
        # Even faster for small sizes:
        # Flatten U, V, W? No.
        # Let's use the loop, it's small (17 iterations).
        Res = np.zeros((num_u, num_v, num_w))
        for r in range(target_rank):
            Res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r], optimize=True)
        return np.sum((Res - T)**2)

    random.seed(42)
    for restart in range(50): # Limit restarts
        if time.time() > deadline - 5:
            break
            
        U = np.random.randint(-1, 2, size=(num_u, target_rank))
        V = np.random.randint(-1, 2, size=(num_v, target_rank))
        W = np.random.randint(-1, 2, size=(num_w, target_rank))
        
        err = calc_error(U, V, W)
        
        # Local search
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            if err < 1e-9:
                return U, V, W
                
            # Pick random variable to change
            mat = random.choice([U, V, W])
            r = random.randint(0, target_rank - 1)
            row = random.randint(0, mat.shape[0] - 1)
            
            old_val = mat[row, r]
            # Try +/- 1 and +/- 0.5
            for delta in [1, -1, 0.5, -0.5]:
                new_val = old_val + delta
                mat[row, r] = new_val
                new_err = calc_error(U, V, W)
                if new_err < err:
                    err = new_err
                else:
                    mat[row, r] = old_val
                    break # If one fails, revert and try next step? Or try other deltas?
            # If no improvement, continue
            
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
