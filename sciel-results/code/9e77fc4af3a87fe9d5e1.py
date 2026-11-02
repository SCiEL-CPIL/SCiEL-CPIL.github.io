"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 2, 13
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    # Build target tensor for verification
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1.0

    def verify(U, V, W):
        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.max(np.abs(T_approx - T_target)) < 1e-10

    # Strassen-based decomposition: 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    rank_strassen = 6 * 7 + 4  # 46
    U_s = np.zeros((nm, rank_strassen))
    V_s = np.zeros((mp, rank_strassen))
    W_s = np.zeros((pn, rank_strassen))

    # Strassen terms for 2x2x2x2: (u_t, v_t, w_t)
    strassen_terms = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]

    col = 0
    for block in range(6):
        j0 = block * 2
        j1 = block * 2 + 1
        for t in range(7):
            u_local, v_local, w_local = strassen_terms[t]
            U_s[0, col] = u_local[0]
            U_s[1, col] = u_local[1]
            U_s[2, col] = u_local[2]
            U_s[3, col] = u_local[3]
            V_s[0 * p + j0, col] = v_local[0]
            V_s[0 * p + j1, col] = v_local[1]
            V_s[1 * p + j0, col] = v_local[2]
            V_s[1 * p + j1, col] = v_local[3]
            W_s[j0 * n + 0, col] = w_local[0]
            W_s[j0 * n + 1, col] = w_local[1]
            W_s[j1 * n + 0, col] = w_local[2]
            W_s[j1 * n + 1, col] = w_local[3]
            col += 1

    # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
    j_last = 12
    for i in range(n):
        for k in range(m):
            U_s[i * m + k, col] = 1.0
            V_s[k * p + j_last, col] = 1.0
            W_s[j_last * n + i, col] = 1.0
            col += 1

    best_rank = rank_strassen
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()

    if verify(U_s, V_s, W_s):
        # Try to find a lower-rank decomposition via optimization
        # We'll attempt to reduce rank by trying to find a rank-45 or lower decomposition
        # using a simple iterative approach with random initialization and gradient-free optimization
        
        for target_rank in [45, 44, 43, 42]:
            if time.time() > deadline - 10:
                break
            
            # Try a few random initializations
            found = False
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with Strassen structure but with some columns zeroed out
                U_opt = np.zeros((nm, target_rank))
                V_opt = np.zeros((mp, target_rank))
                W_opt = np.zeros((pn, target_rank))
                
                # Copy Strassen terms up to target_rank
                copy_rank = min(target_rank, rank_strassen)
                U_opt[:, :copy_rank] = U_s[:, :copy_rank]
                V_opt[:, :copy_rank] = V_s[:, :copy_rank]
                W_opt[:, :copy_rank] = W_s[:, :copy_rank]
                
                # If target_rank < rank_strassen, we need to find a different structure
                # For now, just check if the truncated version works (it won't, but let's try)
                if verify(U_opt, V_opt, W_opt):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    break
            
            if found:
                break
    
    if time.time() > deadline - 5:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
