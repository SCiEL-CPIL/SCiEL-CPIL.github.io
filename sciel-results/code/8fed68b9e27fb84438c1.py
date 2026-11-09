import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _compute_error_np(U, V, W, n, m, p, rank):
    """Vectorized error computation using numpy."""
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.count_nonzero(result != 1)
    return int(error)

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Schoolbook fallback: rank = n*m*p = 56
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

    best_result = None
    best_rank = rank_school

    # Try various ranks from low to high
    for target_rank in [28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 45, 50]:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            break
        result = find_decomposition_fast(n, m, p, target_rank, deadline)
        if result:
            best_result = result
            best_rank = target_rank
            # Found a valid decomposition, try to find even lower
            if target_rank <= 28:
                break

    if best_result:
        return best_result
    return {"U": U_school, "V": V_school, "W": W_school}

def find_decomposition_fast(n, m, p, rank, deadline):
    """
    Fast search for rank-rank decomposition using vectorized error computation.
    """
    total_vars = n * m * rank + m * p * rank + p * n * rank
    
    # Multiple random restarts
    max_restarts = 20
    time_per_restart = (deadline - time.time() - 5) / max_restarts if deadline > time.time() + 5 else 1.0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        restart_deadline = min(deadline - 5, time.time() + max(time_per_restart, 2.0))
        
        # Random initialization with {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        current_error = _compute_error_np(U, V, W, n, m, p, rank)
        if current_error == 0:
            return {"U": U, "V": V, "W": W}
        
        best_error = current_error
        temperature = 5.0
        cooling_rate = 0.995
        no_improve_count = 0
        
        while time.time() < restart_deadline and temperature > 0.05:
            # Pick a random element to change
            mat = random.choice(["U", "V", "W"])
            
            if mat == "U":
                idx = random.randint(0, n*m-1)
                r = random.randint(0, rank-1)
                old_val = U[idx][r]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                U[idx][r] = new_val
                new_error = _compute_error_np(U, V, W, n, m, p, rank)
                
                if new_error < current_error or (temperature > 0.1 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        no_improve_count = 0
                        if best_error == 0:
                            return {"U": U, "V": V, "W": W}
                    else:
                        no_improve_count += 1
                else:
                    U[idx][r] = old_val
                    no_improve_count += 1
            elif mat == "V":
                idx = random.randint(0, m*p-1)
                r = random.randint(0, rank-1)
                old_val = V[idx][r]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                V[idx][r] = new_val
                new_error = _compute_error_np(U, V, W, n, m, p, rank)
                
                if new_error < current_error or (temperature > 0.1 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        no_improve_count = 0
                        if best_error == 0:
                            return {"U": U, "V": V, "W": W}
                    else:
                        no_improve_count += 1
                else:
                    V[idx][r] = old_val
                    no_improve_count += 1
            else:
                idx = random.randint(0, p*n-1)
                r = random.randint(0, rank-1)
                old_val = W[idx][r]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                W[idx][r] = new_val
                new_error = _compute_error_np(U, V, W, n, m, p, rank)
                
                if new_error < current_error or (temperature > 0.1 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        no_improve_count = 0
                        if best_error == 0:
                            return {"U": U, "V": V, "W": W}
                    else:
                        no_improve_count += 1
                else:
                    W[idx][r] = old_val
                    no_improve_count += 1
            
            temperature *= cooling_rate
    
    return None
