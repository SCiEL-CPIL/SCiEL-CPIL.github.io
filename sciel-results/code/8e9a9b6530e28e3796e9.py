import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS
    
    # Build target tensor
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1
    
    def check(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(recon, T)
    
    def error(U, V, W):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return int(np.sum((recon - T) ** 2))
    
    # Rank-26 baseline
    U26 = np.zeros((9, 26), dtype=np.int64)
    V26 = np.zeros((9, 26), dtype=np.int64)
    W26 = np.zeros((9, 26), dtype=np.int64)
    
    col = 0
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    assert check(U26, V26, W26)
    
    best_rank = 26
    best_U, best_V, best_W = U26.copy(), V26.copy(), W26.copy()
    
    # Try to find rank 24 by removing 2 columns and repairing
    all_pairs = list(combinations(range(26), 2))
    rng = np.random.RandomState(42)
    rng.shuffle(all_pairs)
    
    def local_search_r24(U_init, V_init, W_init, max_iters=2000, time_limit=None):
        U_c = U_init.copy().astype(np.int64)
        V_c = V_init.copy().astype(np.int64)
        W_c = W_init.copy().astype(np.int64)
        current_err = error(U_c, V_c, W_c)
        if current_err == 0:
            return U_c, V_c, W_c, 0
        best_err = current_err
        best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
        temperature = 5.0
        cooling = 0.9995
        no_improve = 0
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            mat_choice = rng.randint(3)
            row = rng.randint(9)
            col = rng.randint(24)
            delta = rng.choice([-1, 1])
            if mat_choice == 0:
                U_c[row, col] += delta
            elif mat_choice == 1:
                V_c[row, col] += delta
            else:
                W_c[row, col] += delta
            new_err = error(U_c, V_c, W_c)
            if new_err < current_err or (temperature > 1e-6 and rng.random() < np.exp(-(new_err - current_err) / temperature)):
                current_err = new_err
                no_improve = 0
                if new_err < best_err:
                    best_err = new_err
                    best_Ux, best_Vx, best_Wx = U_c.copy(), V_c.copy(), W_c.copy()
                    if best_err == 0:
                        break
            else:
                if mat_choice == 0:
                    U_c[row, col] -= delta
                elif mat_choice == 1:
                    V_c[row, col] -= delta
                else:
                    W_c[row, col] -= delta
            temperature *= cooling
            no_improve += 1
            if no_improve > 3000 and temperature < 0.5:
                break
        return best_Ux, best_Vx, best_Wx, best_err
    
    for pair in all_pairs:
        if time.time() > deadline - 5:
            break
        cols_to_keep = [c for c in range(26) if c not in pair]
        U_init = np.zeros((9, 24), dtype=np.int64)
        V_init = np.zeros((9, 24), dtype=np.int64)
        W_init = np.zeros((9, 24), dtype=np.int64)
        for new_idx, old_idx in enumerate(cols_to_keep):
            U_init[:, new_idx] = U26[:, old_idx]
            V_init[:, new_idx] = V26[:, old_idx]
            W_init[:, new_idx] = W26[:, old_idx]
        U_sol, V_sol, W_sol, err = local_search_r24(U_init, V_init, W_init, max_iters=1500, time_limit=deadline - 3)
        if err == 0:
            if check(U_sol, V_sol, W_sol):
                best_rank = 24
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
    
    recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if not np.array_equal(recon, T):
        best_U, best_V, best_W = U26, V26, W26
        best_rank = 26
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
