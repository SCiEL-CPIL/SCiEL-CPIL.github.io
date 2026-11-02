import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 105
    
    # Build the matrix multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank
    # Use alternating least squares with random restarts
    for target_rank in range(60, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 2:
                break
            
            np.random.seed(target_rank * 100 + trial)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for it in range(200):
                if time.time() > deadline - 1:
                    break
                
                # Update W
                A_w = np.einsum('ar,br->ab r', U, V).reshape(nm*mp, target_rank)
                T_w = T.reshape(pn, nm*mp).T
                W_new = np.linalg.lstsq(A_w, T_w, rcond=None)[0].T
                W = W_new
                
                # Update U
                A_u = np.einsum('br,cr->bc r', V, W).reshape(mp*pn, target_rank)
                T_u = T.reshape(nm, mp*pn).T
                U_new = np.linalg.lstsq(A_u, T_u, rcond=None)[0].T
                U = U_new
                
                # Update V
                A_v = np.einsum('cr,ar->ac r', W, U).reshape(pn*nm, target_rank)
                T_v = T.reshape(mp, pn*nm).T
                V_new = np.linalg.lstsq(A_v, T_v, rcond=None)[0].T
                V = V_new
            
            # Verify
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_comp - T)
            
            if err < 1e-8:
                # Try rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_v = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.linalg.norm(T_v - T) < 1e-10:
                    def to_list(mat):
                        return [[(int(round(v*2))/2 if abs(v*2-round(v*2))<1e-9 else int(round(v)) if abs(v-round(v))<1e-9 else v) for v in row] for row in mat]
                    best_rank = target_rank
                    best_result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
