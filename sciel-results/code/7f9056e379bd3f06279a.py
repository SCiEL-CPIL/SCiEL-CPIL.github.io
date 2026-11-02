import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    T = np.zeros((size_U, size_V, size_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    max_time = 115  # 5s buffer
    
    lb = max(n*m, m*p, n*p)
    
    def check_and_snap(U, V, W, T):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_recon_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        res_s = np.linalg.norm(T_recon_s - T)
        if res_s < 1e-9:
            return {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
        return None

    for rank in range(lb, 31):
        if time.time() - start_time > max_time:
            break
            
        max_restarts = 20 if rank <= 15 else 10
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
                
            np.random.seed(42 + restart * 100 + rank)
            U = np.random.randn(size_U, rank)
            V = np.random.randn(size_V, rank)
            W = np.random.randn(size_W, rank)
            
            for it in range(100):
                if time.time() - start_time > max_time:
                    break
                    
                # Solve V
                T_flat_V = T.transpose(1, 0, 2).reshape(size_V, -1)
                X = np.zeros((rank, size_U * size_W))
                for r in range(rank):
                    X[r] = (U[:, r] * W[:, r].T).flatten()
                try:
                    V_new, _, _, _ = np.linalg.lstsq(X.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    break
                    
                # Solve W
                T_flat_W = T.transpose(2, 0, 1).reshape(size_W, -1)
                Z = np.zeros((rank, size_U * size_V))
                for r in range(rank):
                    Z[r] = (U[:, r] * V[:, r].T).flatten()
                try:
                    W_new, _, _, _ = np.linalg.lstsq(Z.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    break
                    
                # Solve U
                T_flat_U = T.reshape(size_U, -1)
                Y = np.zeros((rank, size_V * size_W))
                for r in range(rank):
                    Y[r] = (V[:, r] * W[:, r].T).flatten()
                try:
                    U_new, _, _, _ = np.linalg.lstsq(Y.T, T_flat_U.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Check
                if it % 10 == 0:
                    result = check_and_snap(U, V, W, T)
                    if result:
                        return result
                        
    # Fallback
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
