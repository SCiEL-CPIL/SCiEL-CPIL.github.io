import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 110.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_rank_28():
        target_rank = 28
        # We search for integer/half-integer solutions
        # Using random sparse U, V and solving for W
        
        for trial in range(1000):
            if time.time() > deadline:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Bias towards sparse
            U_rand = np.zeros((6, target_rank), dtype=int)
            V_rand = np.zeros((15, target_rank), dtype=int)
            
            # Fill with random -1, 1 with prob 0.3
            mask_u = np.random.rand(6, target_rank) < 0.3
            U_rand[mask_u] = np.random.choice([-1, 1], size=(6, target_rank))[mask_u]
            
            mask_v = np.random.rand(15, target_rank) < 0.3
            V_rand[mask_v] = np.random.choice([-1, 1], size=(15, target_rank))[mask_v]
            
            # Construct linear system for W
            A = np.zeros((30, 10 * target_rank))
            b = np.ones(30)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_rand[i*m+j, :].astype(float)
                        v_row = V_rand[j*p+k, :].astype(float)
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        idx += 1
            
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(10, target_rank)
            
            # Check for integer/half-integer
            W_check = W_sol * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-5):
                W_list = (np.round(W_check) / 2).tolist()
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        return None

    result = try_rank_28()
    if result:
        return result
        
    return schoolbook()
