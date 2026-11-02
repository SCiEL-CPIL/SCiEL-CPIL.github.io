"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as initial best
    
    def is_valid_fast(U, V, W, rank):
        """Check if U, V, W form a valid tensor decomposition using vectorized ops."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            for i in range(n):
                Wc_col_i = Wc[:, i]  # (p,)
                M = Vc * Wc_col_i[None, :]  # (m, p)
                T[i, :, :] += Uc[i, :][:, None] * M
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook
    school = schoolbook()
    if is_valid_fast(school["U"], school["V"], school["W"], n * m * p):
        best_result = school
        best_rank = n * m * p
    else:
        best_result = school
        best_rank = n * m * p
    
    # Try structured search with larger value ranges
    def random_candidate(rank, val_range=4):
        """Generate random integer/half-integer candidate."""
        def rand_val():
            r = random.random()
            if r < 0.35:
                return random.randint(-val_range, val_range)
            elif r < 0.55:
                return random.randint(-2*val_range, 2*val_range) * 0.5
            else:
                return 0
        U = [[rand_val() for _ in range(rank)] for _ in range(n * m)]
        V = [[rand_val() for _ in range(rank)] for _ in range(m * p)]
        W = [[rand_val() for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    # Try random search at various ranks below schoolbook
    for trial in range(40):
        if time.time() > deadline:
            break
        for rank in [88, 84, 80, 76, 72, 68, 64, 60, 56, 52]:
            if time.time() > deadline:
                break
            if rank >= best_rank:
                continue
            U, V, W = random_candidate(rank)
            if is_valid_fast(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END
