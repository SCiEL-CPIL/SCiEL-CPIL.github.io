import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

def _build_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 145
    
    best_result = _build_rank14()
    
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        # Try to find rank 13 using ALS
        target_rank = 13
        n, m, p = 2, 2, 4
        dim_u, dim_v, dim_w = n*m, m*p, p*n
        
        # Create target tensor
        T = np.zeros((dim_u, dim_v, dim_w))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1.0
        
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
            # ALS steps
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                # Minimize || T - sum_r U_r V_r W_r ||^2
                # This is equivalent to solving for each r? No, ALS updates one matrix at a time.
                # Fix V, W. Solve for U.
                # T_{ijk} = sum_r U_{ir} (V_{jr} W_{kr})
                # Let M_{jk}^{(r)} = V_{jr} W_{kr}.
                # This is not a simple linear system for U because of the sum over r.
                # Actually, we can vectorize.
                # Reshape T to (dim_u, dim_v*dim_w).
                # Let T_mat be (dim_u, dim_v*dim_w).
                # Let V_W_mat be (dim_v*dim_w, target_rank) where (V_W_mat)_{(j,k), r} = V_{jr} W_{kr}.
                # Then T_mat = U @ V_W_mat^T.
                # So U = T_mat @ (V_W_mat^T)^+ (pseudo-inverse).
                
                V_W_flat = (V[:, :, None] * W[None, :, :]).reshape(dim_v * dim_w, target_rank)
                T_mat = T.reshape(dim_u, dim_v * dim_w)
                
                # Solve U @ V_W_flat^T = T_mat
                # U = T_mat @ pinv(V_W_flat^T)
                U_new, _, _, _ = np.linalg.lstsq(V_W_flat.T, T_mat.T, rcond=None)
                U = U_new.T
                
                # Update V
                # T_{ijk} = sum_r (U_{ir} W_{kr}) V_{jr}
                # Let U_W_mat be (dim_u*dim_w, target_rank) where (U_W_mat)_{(i,k), r} = U_{ir} W_{kr}.
                # T_mat_V = T.reshape(dim_v, dim_u*dim_w).T
                U_W_flat = (U[:, :, None] * W[None, :, :]).reshape(dim_u * dim_w, target_rank)
                T_mat_V = T.reshape(dim_v, dim_u * dim_w).T
                
                V_new, _, _, _ = np.linalg.lstsq(U_W_flat.T, T_mat_V.T, rcond=None)
                V = V_new.T
                
                # Update W
                # T_{ijk} = sum_r (U_{ir} V_{jr}) W_{kr}
                # Let U_V_mat be (dim_u*dim_v, target_rank) where (U_V_mat)_{(i,j), r} = U_{ir} V_{jr}.
                U_V_flat = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, target_rank)
                T_mat_W = T.reshape(dim_w, dim_u * dim_v).T
                
                W_new, _, _, _ = np.linalg.lstsq(U_V_flat.T, T_mat_W.T, rcond=None)
                W = W_new.T
                
                # Check error
                error = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
                if error < 1e-9:
                    # Verify
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
