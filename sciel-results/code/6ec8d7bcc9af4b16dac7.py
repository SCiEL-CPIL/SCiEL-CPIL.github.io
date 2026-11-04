import numpy as np
import random
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution as fallback
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
    field_elements = np.arange(field_size)
    
    # Precompute indices for tensor elements
    # T[i, j, k] corresponds to U[i*m+j], V[j*p+k], W[k*n+i]
    # We want sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    
    # We will use NumPy to speed up error calculation and updates.
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    def get_error(U, V, W, R):
        # Compute product tensor P[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # This is a batched matmul or einsum.
        # Let's reshape U, V, W to 3D to use einsum.
        # U_reshaped[i, j, c] = U[i*m+j, c]
        # V_reshaped[j, k, c] = V[j*p+k, c]
        # W_reshaped[k, i, c] = W[k*n+i, c]
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Product at (i, j, k) is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note: W index is k*n+i, so W_3d[k, i, c] is correct.
        
        # einsum: 'ijc,jkc,kic->ijk'
        P = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        P = P % field_size
        
        # Target is all 1s
        error = np.sum(P != 1)
        return int(error)

    start_time = time.time()
    max_time = 115

    # We try ranks from a lower bound upwards.
    # Theoretical lower bound for M_{4,5,5} is not 28 necessarily, but let's try low ranks.
    # Known: Rank(M_{2,2,2})=7. Rank(M_{3,3,3})<=23?
    # Let's try ranks 28 to 50.
    
    for R in range(28, 60):
        if time.time() - start_time > max_time:
            break
            
        # Initialize
        # Random initialization
        U = np.random.choice(field_elements, size=(n * m, R)).astype(int)
        V = np.random.choice(field_elements, size=(m * p, R)).astype(int)
        W = np.random.choice(field_elements, size=(p * n, R)).astype(int)
        
        current_err = get_error(U, V, W, R)
        
        if current_err == 0:
            best_solution = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = R
            break
            
        # Local Search / Simulated Annealing
        T = 100.0
        no_improve = 0
        max_iter = 5000
        
        while current_err > 0 and no_improve < 100 and time.time() - start_time < max_time:
            # Pick a random update
            mat = random.choice([0, 1, 2])
            
            if mat == 0:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, R - 1)
                old_val = U[idx, col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                U[idx, col] = new_val
                new_err = get_error(U, V, W, R)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_err - current_err
                    if random.random() < np.exp(-delta / T):
                        current_err = new_err
                        no_improve = 0
                    else:
                        U[idx, col] = old_val
                        no_improve += 1
            elif mat == 1:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, R - 1)
                old_val = V[idx, col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                V[idx, col] = new_val
                new_err = get_error(U, V, W, R)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve = 0
                else:
                    delta = new_err - current_err
                    if random.random() < np.exp(-delta / T):
                        current_err = new_err
                        no_improve = 0
                    else:
                        V[idx, col] = old_val
                        no_improve += 1
            else:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, R - 1)
                old_val = W[idx, col]
                new_val = random.choice(field_elements)
                if new_val == old_val:
                    continue
                W[idx, col] = new_val
                new_err = get_error(U, V, W, R)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve = 0
                else:
                    delta = new_err - current_err
                    if random.random() < np.exp(-delta / T):
                        current_err = new_err
                        no_improve = 0
                    else:
                        W[idx, col] = old_val
                        no_improve += 1
                        
            # Cool down
            T *= 0.99
            if T < 0.1:
                T = 0.1
                
            if current_err == 0:
                break
                
        if current_err == 0:
            best_solution = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = R
            break

    return best_solution
