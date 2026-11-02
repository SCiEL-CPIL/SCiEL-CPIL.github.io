import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Build the target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0

    # Unfolds for ALS
    T_1 = T.reshape(num_u, num_v * num_w)
    T_2 = T.transpose(1, 2, 0).reshape(num_v, num_w * num_u)
    T_3 = T.transpose(2, 0, 1).reshape(num_w, num_u * num_v)
    
    def check_solution(U, V, W):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        return np.linalg.norm(diff) < 1e-5

    def try_rank_als(R, max_restarts=50, max_iters=50):
        start_time = time.time()
        
        for restart in range(max_restarts):
            if time.time() > 140.0:
                return None
            
            np.random.seed(42 + R * 100 + restart)
            U = np.random.randn(num_u, R)
            V = np.random.randn(num_v, R)
            W = np.random.randn(num_w, R)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > 140.0:
                    return None
                
                # Update U
                M_u = np.einsum('vr, wr -> vw r', V, W).reshape(num_v * num_w, R)
                try:
                    U, _, _, _ = np.linalg.lstsq(M_u.T, T_1.T, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Update V
                N_v = np.einsum('wr, ur -> wu r', W, U).reshape(num_w * num_u, R)
                try:
                    V, _, _, _ = np.linalg.lstsq(N_v.T, T_2.T, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                    
                # Update W
                P_w = np.einsum('ur, vr -> uv r', U, V).reshape(num_u * num_v, R)
                try:
                    W, _, _, _ = np.linalg.lstsq(P_w.T, T_3.T, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check error
                T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
                err = np.linalg.norm(T_calc - T)
                
                if err < 1e-8:
                    # Try to round
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    if check_solution(U_round, V_round, W_round):
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
                
        return None

    # Try ranks
    for R in range(21, 64):
        res = try_rank_als(R)
        if res is not None:
            return res
            
    # Fallback: Schoolbook
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
