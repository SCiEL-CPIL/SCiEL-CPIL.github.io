import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Current known solution rank 28
    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U_known[X_rows[0]][col_offset + 0] += 1
        U_known[X_rows[3]][col_offset + 0] += 1
        V_known[Y_rows[0]][col_offset + 0] += 1
        V_known[Y_rows[3]][col_offset + 0] += 1
        W_known[Z_rows[0]][col_offset + 0] += 1
        W_known[Z_rows[3]][col_offset + 0] += 1
        
        U_known[X_rows[2]][col_offset + 1] += 1
        U_known[X_rows[3]][col_offset + 1] += 1
        V_known[Y_rows[0]][col_offset + 1] += 1
        W_known[Z_rows[2]][col_offset + 1] += 1
        W_known[Z_rows[3]][col_offset + 1] -= 1
        
        U_known[X_rows[0]][col_offset + 2] += 1
        V_known[Y_rows[1]][col_offset + 2] += 1
        V_known[Y_rows[3]][col_offset + 2] -= 1
        W_known[Z_rows[1]][col_offset + 2] += 1
        W_known[Z_rows[3]][col_offset + 2] += 1
        
        U_known[X_rows[3]][col_offset + 3] += 1
        V_known[Y_rows[2]][col_offset + 3] += 1
        V_known[Y_rows[0]][col_offset + 3] -= 1
        W_known[Z_rows[0]][col_offset + 3] += 1
        W_known[Z_rows[2]][col_offset + 3] += 1
        
        U_known[X_rows[0]][col_offset + 4] += 1
        U_known[X_rows[1]][col_offset + 4] += 1
        V_known[Y_rows[3]][col_offset + 4] += 1
        W_known[Z_rows[0]][col_offset + 4] -= 1
        W_known[Z_rows[1]][col_offset + 4] += 1
        
        U_known[X_rows[2]][col_offset + 5] += 1
        U_known[X_rows[0]][col_offset + 5] -= 1
        V_known[Y_rows[0]][col_offset + 5] += 1
        V_known[Y_rows[1]][col_offset + 5] += 1
        W_known[Z_rows[3]][col_offset + 5] += 1
        
        U_known[X_rows[1]][col_offset + 6] += 1
        U_known[X_rows[3]][col_offset + 6] -= 1
        V_known[Y_rows[2]][col_offset + 6] += 1
        V_known[Y_rows[3]][col_offset + 6] += 1
        W_known[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    X2 = [2, 3, 6, 7]
    Y11 = [0, 1, 4, 5]
    Y12 = [2, 3, 6, 7]
    Y21 = [8, 9, 12, 13]
    Y22 = [10, 11, 14, 15]
    Z11 = [0, 2, 1, 3]
    Z12 = [4, 6, 5, 7]
    
    add_block(0, X1, Y11, Z11)
    add_block(7, X2, Y21, Z11)
    add_block(14, X1, Y12, Z12)
    add_block(21, X2, Y22, Z12)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110

    # Attempt to find rank 27 via randomized search
    # We assume the solution has integer entries in {-1, 0, 1}
    # and is sparse.
    
    def fast_check(U, V, W, rank):
        # U: (8, rank), V: (16, rank), W: (8, rank)
        # T_est[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can compute this as: T_est = U @ (V * W.T).T ? No.
        # T_est[i, j, k] = sum_r U[i,r] * (V[j,r] * W[k,r])
        # Let M = V * W.T (shape 16 x 8). Then M[j,k] = sum_r V[j,r] W[k,r] is NOT what we want.
        # We need sum_r U[i,r] V[j,r] W[k,r].
        # This is a 3-way contraction.
        # Vectorized: 
        # T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.allclose(T_est, T, atol=1e-9)

    # Random search strategy
    # Generate random sparse vectors.
    # Based on the known solution, each column of U has ~2 non-zeros.
    # Each column of V has ~2 non-zeros.
    # Each column of W has ~2 non-zeros.
    
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        U = np.zeros((8, 27), dtype=int)
        V = np.zeros((16, 27), dtype=int)
        W = np.zeros((8, 27), dtype=int)
        
        # Randomly assign non-zero entries
        # Heuristic: Each rank-1 term should contribute to a specific subset of the tensor.
        # We will just sample random sparse patterns and check.
        
        for r in range(27):
            # U column
            nz_u = np.random.choice(8, 2, replace=False)
            U[nz_u, r] = np.random.choice([-1, 1], 2)
            
            # V column
            nz_v = np.random.choice(16, 2, replace=False)
            V[nz_v, r] = np.random.choice([-1, 1], 2)
            
            # W column
            nz_w = np.random.choice(8, 2, replace=False)
            W[nz_w, r] = np.random.choice([-1, 1], 2)
            
        if fast_check(U, V, W, 27):
            best_rank = 27
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            break

    return {"U": best_U, "V": best_V, "W": best_W}
