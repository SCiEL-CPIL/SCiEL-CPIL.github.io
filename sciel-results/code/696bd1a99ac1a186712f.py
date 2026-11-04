import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Uses Simulated Annealing with incremental cost updates.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify a candidate decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Precompute indices for fast access
    # U indices: (i, j, r) -> flat index in U
    # V indices: (j, k, r) -> flat index in V
    # W indices: (k, i, r) -> flat index in W
    
    def sa_search(target_rank, max_iterations=50000):
        # Initialize with random small integers
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        W = np.random.randint(-2, 3, size=(p*n, target_rank))
        
        # Calculate initial tensor values T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        
        current_cost = np.sum((T - 1.0) ** 2)
        
        T_temp = 10.0
        T_min = 0.01
        cooling_rate = 0.9999
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            var_type = random.choice(['U', 'V', 'W'])
            
            if var_type == 'U':
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                r = random.randint(0, target_rank-1)
                u_idx = i*m + j
                old_val = U[u_idx, r]
                new_val = old_val + random.choice([-1, 1])
                if new_val < -3 or new_val > 3: continue
                
                # Update cost for all k
                delta_cost = 0
                for k in range(p):
                    old_T = T[i,j,k]
                    # T[i,j,k] += (new_val - old_val) * V[j,k,r] * W[k,i,r]
                    delta_T = (new_val - old_val) * V[j*p+k, r] * W[k*n+i, r]
                    new_T = old_T + delta_T
                    delta_cost += (new_T - 1.0)**2 - (old_T - 1.0)**2
                    T[i,j,k] = new_T
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T_temp):
                    U[u_idx, r] = new_val
                    current_cost += delta_cost
                    if current_cost < 1e-12:
                        if verify(U, V, W):
                            return U.tolist(), V.tolist(), W.tolist()
                else:
                    # Revert T changes
                    for k in range(p):
                        old_T = T[i,j,k]
                        delta_T = (new_val - old_val) * V[j*p+k, r] * W[k*n+i, r]
                        T[i,j,k] = old_T - delta_T
                    # U was not changed
            
            elif var_type == 'V':
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                r = random.randint(0, target_rank-1)
                v_idx = j*p + k
                old_val = V[v_idx, r]
                new_val = old_val + random.choice([-1, 1])
                if new_val < -3 or new_val > 3: continue
                
                delta_cost = 0
                for i in range(n):
                    old_T = T[i,j,k]
                    delta_T = (new_val - old_val) * U[i*m+j, r] * W[k*n+i, r]
                    new_T = old_T + delta_T
                    delta_cost += (new_T - 1.0)**2 - (old_T - 1.0)**2
                    T[i,j,k] = new_T
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T_temp):
                    V[v_idx, r] = new_val
                    current_cost += delta_cost
                    if current_cost < 1e-12:
                        if verify(U, V, W):
                            return U.tolist(), V.tolist(), W.tolist()
                else:
                    for i in range(n):
                        old_T = T[i,j,k]
                        delta_T = (new_val - old_val) * U[i*m+j, r] * W[k*n+i, r]
                        T[i,j,k] = old_T - delta_T
            
            else: # W
                k = random.randint(0, p-1)
                i = random.randint(0, n-1)
                r = random.randint(0, target_rank-1)
                w_idx = k*n + i
                old_val = W[w_idx, r]
                new_val = old_val + random.choice([-1, 1])
                if new_val < -3 or new_val > 3: continue
                
                delta_cost = 0
                for j in range(m):
                    old_T = T[i,j,k]
                    delta_T = (new_val - old_val) * U[i*m+j, r] * V[j*p+k, r]
                    new_T = old_T + delta_T
                    delta_cost += (new_T - 1.0)**2 - (old_T - 1.0)**2
                    T[i,j,k] = new_T
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T_temp):
                    W[w_idx, r] = new_val
                    current_cost += delta_cost
                    if current_cost < 1e-12:
                        if verify(U, V, W):
                            return U.tolist(), V.tolist(), W.tolist()
                else:
                    for j in range(m):
                        old_T = T[i,j,k]
                        delta_T = (new_val - old_val) * U[i*m+j, r] * V[j*p+k, r]
                        T[i,j,k] = old_T - delta_T
            
            T_temp *= cooling_rate
            if T_temp < T_min:
                T_temp = T_min
        
        return None

    random.seed(42)
    
    # Try ranks from low to high
    # Lower bound is 20
    for target_rank in range(20, 45):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Increase iterations for lower ranks as they are harder
        max_iter = 20000 if target_rank < 30 else 5000
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            result = sa_search(target_rank, max_iterations=max_iter)
            if result:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break

    if best_rank < n * m * p:
        return best
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
