import numpy as np
import random
import time

def construct():
    n, m, p = 3, 3, 6
    
    # Try to find rank 13 decomposition for 3x3x2
    # U: 9x13, V: 6x13, W: 6x13
    # Equations: sum_l U[i*m+j, l] V[j*p'+k, l] W[k*n'+i, l] = 1 for i,j,k in 3x3x2
    # Here p'=2, n'=3.
    # Indices:
    # U index: i*3 + j
    # V index: j*2 + k
    # W index: k*3 + i
    
    rank_target = 13
    U_shape = (9, rank_target)
    V_shape = (6, rank_target)
    W_shape = (6, rank_target)
    
    # Indices for the 18 equations
    eqs = []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                eqs.append((i*3+j, j*2+k, k*3+i))
                
    # Initialize random
    U = np.random.choice([-1, 0, 1], size=U_shape)
    V = np.random.choice([-1, 0, 1], size=V_shape)
    W = np.random.choice([-1, 0, 1], size=W_shape)
    
    def calc_error(U, V, W):
        errors = []
        for ui, vi, wi in eqs:
            val = np.sum(U[ui] * V[vi] * W[wi])
            errors.append(val - 1)
        return errors

    start_time = time.time()
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple local search
    # We have 273 variables.
    # We can iterate over variables and flip them if it reduces the sum of squares of errors.
    
    # To speed up, precompute contributions.
    # This is complex. Let's just do a random restart search with a few iterations.
    
    for _ in range(100): # 100 restarts
        U = np.random.choice([-1, 0, 1], size=U_shape)
        V = np.random.choice([-1, 0, 1], size=V_shape)
        W = np.random.choice([-1, 0, 1], size=W_shape)
        
        for it in range(1000):
            if time.time() - start_time > 140:
                break
            # Pick a random variable
            var_type = random.choice(['U', 'V', 'W'])
            if var_type == 'U':
                idx = random.randint(0, 8)
                col = random.randint(0, 12)
                old = U[idx, col]
                new = random.choice([-1, 0, 1])
                if new == old: continue
                U[idx, col] = new
                err = np.sum(np.square(calc_error(U, V, W)))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err == 0:
                        break
                else:
                    U[idx, col] = old
            elif var_type == 'V':
                idx = random.randint(0, 5)
                col = random.randint(0, 12)
                old = V[idx, col]
                new = random.choice([-1, 0, 1])
                if new == old: continue
                V[idx, col] = new
                err = np.sum(np.square(calc_error(U, V, W)))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err == 0:
                        break
                else:
                    V[idx, col] = old
            else:
                idx = random.randint(0, 5)
                col = random.randint(0, 12)
                old = W[idx, col]
                new = random.choice([-1, 0, 1])
                if new == old: continue
                W[idx, col] = new
                err = np.sum(np.square(calc_error(U, V, W)))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err == 0:
                        break
                else:
                    W[idx, col] = old
        if best_err == 0:
            break

    if best_err == 0:
        # Construct the 3x3x6 decomposition
        # It is the direct sum of three 3x3x2 decompositions
        # U_total: 9x39
        # V_total: 18x39 (since p=6, m*p = 3*6=18)
        # W_total: 18x39 (since p*n = 6*3=18)
        
        U_total = np.zeros((9, 39))
        V_total = np.zeros((18, 39))
        W_total = np.zeros((18, 39))
        
        for block in range(3):
            # U is same for all blocks
            U_total[:, block*13:(block+1)*13] = best_U
            
            # V: rows are j*6 + k? No, V index is j*p + k.
            # In 3x3x2, V index was j*2 + k.
            # In 3x3x6, V index is j*6 + k.
            # For block b, k ranges from 2b to 2b+1.
            # So V[j*6 + 2b + k', l] = V_2x2[j*2 + k', l]
            for j in range(3):
                for k_prime in range(2):
                    v_idx_2x2 = j*2 + k_prime
                    v_idx_6x6 = j*6 + (2*block + k_prime)
                    V_total[v_idx_6x6, block*13:(block+1)*13] = best_V[v_idx_2x2]
            
            # W: rows are k*n + i.
            # In 3x3x2, W index was k*3 + i.
            # In 3x3x6, W index is k*3 + i.
            # For block b, k ranges from 2b to 2b+1.
            # So W[(2b+k')*3 + i, l] = W_2x2[k'*3 + i, l]
            for k_prime in range(2):
                for i in range(3):
                    w_idx_2x2 = k_prime*3 + i
                    w_idx_6x6 = (2*block + k_prime)*3 + i
                    W_total[w_idx_6x6, block*13:(block+1)*13] = best_W[w_idx_2x2]
        
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook rank 54
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

