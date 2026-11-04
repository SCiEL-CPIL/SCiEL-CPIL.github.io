"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the 4x4x9 multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    def try_rank(T, rank, max_als_iters=40, max_restarts=5, seed=42):
        I, J, K = T.shape
        best_result = None
        best_err = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            rng = np.random.RandomState(seed + restart)
            U = rng.randn(I, rank)
            V = rng.randn(J, rank)
            W = rng.randn(K, rank)
            
            for it in range(max_als_iters):
                M_U = np.zeros((rank, J * K))
                for t in range(rank):
                    M_U[t, :] = (V[:, t] * W[:, t].reshape(1, -1)).flatten()
                U, _, _, _ = np.linalg.lstsq(M_U.T, T.reshape(I, J*K), rcond=None)
                
                M_V = np.zeros((rank, I * K))
                for t in range(rank):
                    M_V[t, :] = (U[:, t] * W[:, t].reshape(1, -1)).flatten()
                V, _, _, _ = np.linalg.lstsq(M_V.T, np.einsum('ijk->jik', T).reshape(J, I*K), rcond=None)
                
                M_W = np.zeros((rank, I * J))
                for t in range(rank):
                    M_W[t, :] = (U[:, t] * V[:, t].reshape(1, -1)).flatten()
                W, _, _, _ = np.linalg.lstsq(M_W.T, np.einsum('ijk->kij', T).reshape(K, I*J), rcond=None)
            
            T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            if err < best_err:
                best_err = err
                best_result = (U, V, W)
            if err < 1e-6:
                break
        
        return best_result, best_err
    
    # Try ranks in descending order from schoolbook
    best_decomp = None
    for rank in [144, 140, 135, 130, 125, 120, 115, 110, 105, 100]:
        if time.time() > deadline - 20:
            break
        result, err = try_rank(T, rank, max_als_iters=30, max_restarts=4, seed=rank)
        if result is not None and err < 1e-5:
            best_decomp = result
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        for arr in [U_list, V_list, W_list]:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j] * 2) / 2
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook fallback
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

