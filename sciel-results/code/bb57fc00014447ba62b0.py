import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][r] = 1
                V_school[j * p + k][r] = 1
                W_school[k * n + i][r] = 1
                r += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Search for rank 25
    target_rank = 25
    
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Random U and W with small integers
        U_rand = np.random.randint(0, 2, (n * m, target_rank))
        W_rand = np.random.randint(0, 2, (p * n, target_rank))
        
        # Build linear system for V
        # Eq: sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
        num_eqs = n * m * p
        num_unknowns = m * p * target_rank
        
        A_mat = np.zeros((num_eqs, num_unknowns))
        b_vec = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(target_rank):
                        col_idx = (j * p + k) * target_rank + r
                        A_mat[eq_idx, col_idx] = U_rand[i * m + j, r] * W_rand[k * n + i, r]
                    eq_idx += 1
        
        try:
            V_flat, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
            
        V_rand = V_flat.reshape(m * p, target_rank)
        
        # Snap to integers
        U_snap = np.rint(U_rand)
        V_snap = np.rint(V_rand)
        W_snap = np.rint(W_rand)
        
        # Verify
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_result = {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            break
        
        # Try half-integers if integers failed?
        # Snap to half-integers
        U_snap2 = np.rint(U_rand * 2) / 2.0
        V_snap2 = np.rint(V_rand * 2) / 2.0
        W_snap2 = np.rint(W_rand * 2) / 2.0
        
        valid2 = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_snap2[i * m + j, r] * V_snap2[j * p + k, r] * W_snap2[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        valid2 = False
                        break
                if not valid2:
                    break
            if not valid2:
                break
                
        if valid2:
            best_rank = target_rank
            best_result = {
                "U": U_snap2.tolist(),
                "V": V_snap2.tolist(),
                "W": W_snap2.tolist()
            }
            break

    return best_result
