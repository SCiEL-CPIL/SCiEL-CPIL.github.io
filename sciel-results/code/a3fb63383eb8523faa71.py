"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    dim_u = n * m # 6
    dim_v = m * p # 12
    dim_w = p * n # 8
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def compute_error(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((P - T) ** 2)
        
    def verify(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(P, T, atol=1e-9)
        
    def schoolbook():
        r = n*m*p
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    best_result = schoolbook()
    best_rank = n*m*p
    
    # Try ranks 14 to 23
    for rank in range(14, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Candidates
        candidates = np.array([-1, 0, 1, 0.5, -0.5])
        
        found = False
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 123 + rank)
            # Random initialization
            U = np.random.choice(candidates, size=(dim_u, rank))
            V = np.random.choice(candidates, size=(dim_v, rank))
            W = np.random.choice(candidates, size=(dim_w, rank))
            
            cur_err = compute_error(U, V, W)
            
            # Local Search
            max_iters = 3000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                if verify(U, V, W):
                    found = True
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
                    
                # Randomly pick a matrix and entry to change
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                    old_val = U[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val: continue
                    U[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        U[idx, t] = old_val
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                    old_val = V[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val: continue
                    V[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        V[idx, t] = old_val
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    old_val = W[idx, t]
                    new_val = np.random.choice(candidates)
                    if new_val == old_val: continue
                    W[idx, t] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < cur_err:
                        cur_err = new_err
                    else:
                        W[idx, t] = old_val
                        
            if found:
                break
                
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
