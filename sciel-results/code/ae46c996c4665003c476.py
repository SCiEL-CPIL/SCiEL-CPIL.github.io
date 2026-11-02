# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 3
    dim = 9
    T3_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T3_nonzero[(a, b, c)] = 1.0

    U3 = None
    V3 = None
    W3 = None
    target_rank = 23
    
    # Pre-compute matrices for faster least squares
    # We will try to find U, V, W with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    # Scaling by 2, we look for integers in {-2, -1, 0, 1, 2}
    
    # To improve search, we can use a structured initialization
    # e.g., sparse matrices
    
    for attempt in range(2000): # Increased attempts
        if time.time() > deadline - 10: # Leave 10s for construction and verification
            break
            
        # Random sparse initialization
        # Density 0.5
        U_rand = np.zeros((dim, target_rank))
        V_rand = np.zeros((dim, target_rank))
        
        for _ in range(dim * target_rank // 2):
            r = np.random.randint(0, dim)
            c = np.random.randint(0, target_rank)
            U_rand[r, c] = np.random.choice([-1, 1])
            
        for _ in range(dim * target_rank // 2):
            r = np.random.randint(0, dim)
            c = np.random.randint(0, target_rank)
            V_rand[r, c] = np.random.choice([-1, 1])

        W_rand = np.zeros((dim, target_rank))
        valid = True
        
        for c in range(dim):
            M_c = np.zeros((dim * dim, target_rank))
            b_c = np.zeros(dim * dim)
            
            idx = 0
            for a in range(dim):
                for b in range(dim):
                    M_c[idx, :] = U_rand[a, :] * V_rand[b, :]
                    if (a, b, c) in T3_nonzero:
                        b_c[idx] = 1.0
                    else:
                        b_c[idx] = 0.0
                    idx += 1
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
            except:
                valid = False
                break
            
            # Check for half-integers (scale by 2)
            w_doubled = w * 2
            closest_ints = np.round(w_doubled)
            error = np.max(np.abs(w_doubled - closest_ints))
            
            if error < 1e-9:
                W_rand[c, :] = w
            else:
                valid = False
                break
        
        if valid:
            # Verify
            is_valid = True
            for (a, b, c), val in T3_nonzero.items():
                s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                if abs(s - val) > 1e-9:
                    is_valid = False
                    break
            
            if is_valid:
                # Check zero entries
                for _ in range(100):
                    a = np.random.randint(0, dim)
                    b = np.random.randint(0, dim)
                    c = np.random.randint(0, dim)
                    if (a, b, c) not in T3_nonzero:
                        s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                        if abs(s) > 1e-9:
                            is_valid = False
                            break
                
                if is_valid:
                    U3 = U_rand
                    V3 = V_rand
                    W3 = W_rand
                    break

    if U3 is not None:
        U6 = np.zeros((9, 46))
        V6 = np.zeros((18, 46))
        W6 = np.zeros((18, 46))
        
        U6[:, 0:23] = U3
        V6[0:9, 0:23] = V3
        W6[0:9, 0:23] = W3
        
        U6[:, 23:46] = U3
        V6[9:18, 23:46] = V3
        W6[9:18, 23:46] = W3
        
        U_list = [[int(x) for x in row] for row in U6.tolist()]
        V_list = [[int(x) for x in row] for row in V6.tolist()]
        W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W6.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        n, m, p = 3, 3, 6
        dim_A = 9
        dim_B = 18
        dim_C = 18
        rank = 54
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
