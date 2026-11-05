import numpy as np
import random
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    field_size = 7
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    indices = np.array(indices)
    
    def tensor_error_np(U, V, W):
        u_vals = U[indices[:, 0]]
        v_vals = V[indices[:, 1]]
        w_vals = W[indices[:, 2]]
        prod = u_vals * v_vals * w_vals
        sums = prod.sum(axis=1) % field_size
        mismatches = np.sum(sums != 1)
        return int(mismatches)
    
    start_time = time.time()
    max_time = 110
    
    for R in range(25, best_rank):
        if time.time() - start_time > max_time:
            break
        
        found = False
        max_restarts = 8 if R <= 30 else 5 if R <= 40 else 3
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            U = np.random.randint(0, field_size, (n * m, R))
            V = np.random.randint(0, field_size, (m * p, R))
            W = np.random.randint(0, field_size, (p * n, R))
            
            current_err = tensor_error_np(U, V, W)
            no_improve = 0
            
            while current_err > 0 and no_improve < 150 and time.time() - start_time < max_time:
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    old_val = U[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    U[idx, col] = new_val
                    new_err = tensor_error_np(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        U[idx, col] = old_val
                        no_improve += 1
                elif mat == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    old_val = V[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    V[idx, col] = new_val
                    new_err = tensor_error_np(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        V[idx, col] = old_val
                        no_improve += 1
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    old_val = W[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    W[idx, col] = new_val
                    new_err = tensor_error_np(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        W[idx, col] = old_val
                        no_improve += 1
                
                if current_err == 0:
                    break
            
            if current_err == 0:
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    return best_solution
