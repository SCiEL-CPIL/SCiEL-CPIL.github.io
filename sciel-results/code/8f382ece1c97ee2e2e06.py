import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Construct target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) match, else 0
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    best_solution = None
    best_rank = n * m * p
    
    # Fallback schoolbook
    U_fb = np.zeros((N_U, best_rank))
    V_fb = np.zeros((N_V, best_rank))
    W_fb = np.zeros((N_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try ranks from low to high
    for rank in range(19, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        for restart in range(200):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank * 100 + restart)
            
            # Initialize
            U = np.random.randn(N_U, rank)
            V = np.random.randn(N_V, rank)
            W = np.random.randn(N_W, rank)
            
            # ALS Iterations
            for it in range(100):
                # Update U
                # T_mat: (N_U, N_V * N_W)
                T_mat_U = T.reshape(N_U, N_V * N_W)
                # M_VW: (N_V * N_W, rank)
                M_VW = np.zeros((N_V * N_W, rank))
                for v in range(N_V):
                    for w in range(N_W):
                        M_VW[v * N_W + w, :] = V[v, :] * W[w, :]
                try:
                    U, _, _, _ = np.linalg.lstsq(M_VW, T_mat_U.T, rcond=None)
                    U = U.T
                except np.linalg.LinAlgError:
                    pass
                    
                # Update V
                T_mat_V = T.reshape(N_V, N_U * N_W)
                M_UW = np.zeros((N_U * N_W, rank))
                for u in range(N_U):
                    for w in range(N_W):
                        M_UW[u * N_W + w, :] = U[u, :] * W[w, :]
                try:
                    V, _, _, _ = np.linalg.lstsq(M_UW, T_mat_V.T, rcond=None)
                    V = V.T
                except np.linalg.LinAlgError:
                    pass
                    
                # Update W
                T_mat_W = T.reshape(N_W, N_U * N_V)
                M_UV = np.zeros((N_U * N_V, rank))
                for u in range(N_U):
                    for v in range(N_V):
                        M_UV[u * N_V + v, :] = U[u, :] * V[v, :]
                try:
                    W, _, _, _ = np.linalg.lstsq(M_UV, T_mat_W.T, rcond=None)
                    W = W.T
                except np.linalg.LinAlgError:
                    pass
            
            # Round to nearest 0.5
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Check exactness
            # Reconstruct
            T_recon = np.zeros((N_U, N_V, N_W))
            for c in range(rank):
                T_recon += np.outer(np.outer(U_round[:, c], V_round[:, c]), W_round[:, c])
                
            if np.allclose(T_recon, T, atol=1e-6):
                # Valid solution found
                # Convert to lists of floats (or ints if integer)
                U_list = [[float(x) for x in row] for row in U_round]
                V_list = [[float(x) for x in row] for row in V_round]
                W_list = [[float(x) for x in row] for row in W_round]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better solution found, return schoolbook
    U_list = [[float(x) for x in row] for row in U_fb]
    V_list = [[float(x) for x in row] for row in V_fb]
    W_list = [[float(x) for x in row] for row in W_fb]
    return {"U": U_list, "V": V_list, "W": W_list}

