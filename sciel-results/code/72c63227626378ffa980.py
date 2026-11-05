import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Target Tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.int64)
        V = np.zeros((dim_v, rank), dtype=np.int64)
        W = np.zeros((dim_w, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Precompute tensor slices for fast error calculation
    # T_hat[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Error = ||T - T_hat||^2 = ||T||^2 + ||T_hat||^2 - 2 <T, T_hat>
    # <T, T_hat> = sum_{a,b,c} T[a,b,c] T_hat[a,b,c]
    # Since T is 0/1, this is sum_{(a,b,c) in supp} T_hat[a,b,c]
    
    # We can compute T_hat[a,b,c] efficiently.
    # For a fixed c, T_hat[:, :, c] = sum_r (U[:, r] V[:, r].T) W[c, r]
    # This is a linear combination of rank-1 matrices U[:, r] V[:, r].T.
    
    # Let's precompute the list of non-zero indices of T
    t_indices = np.argwhere(T != 0)
    # t_indices shape: (100, 3)
    
    def compute_error(U, V, W):
        # U: (20, R), V: (25, R), W: (20, R)
        # We need sum_{(a,b,c) in supp} (U[a,r] V[b,r] W[c,r]) summed over r
        # This is equivalent to:
        # For each (a,b,c) in supp, compute dot product of U[a,:], V[b,:], W[c,:]
        # But that's O(R) per index, 100 * R.
        # Total O(100 R).
        # R ~ 60-100. So ~10,000 ops. Very fast.
        
        # Even faster:
        # Let M = U @ V.T (20 x 25). M[a,b] = sum_r U[a,r] V[b,r]
        # Then T_hat[a,b,c] = sum_r (sum_r' ... no.
        # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is not simply M @ W.
        # It is the element-wise product of M and ... no.
        # It is the contraction of U, V, W.
        
        # Let's use the property that T is sparse.
        # sum_{(a,b,c) in supp} T_hat[a,b,c]
        # = sum_{(a,b,c) in supp} sum_r U[a,r] V[b,r] W[c,r]
        # = sum_r ( sum_{(a,b,c) in supp} U[a,r] V[b,r] W[c,r] )
        
        # We can precompute for each r? No, U, V, W change.
        
        # Let's just compute it directly using numpy vectorization on the indices.
        # U_idx = U[t_indices[:, 0], :] -> (100, R)
        # V_idx = V[t_indices[:, 1], :] -> (100, R)
        # W_idx = W[t_indices[:, 2], :] -> (100, R)
        # T_hat_vals = np.sum(U_idx * V_idx * W_idx, axis=1) -> (100,)
        # inner_product = np.sum(T_hat_vals)
        
        # Norm of T is 100 (since 100 ones).
        # Norm of T_hat squared:
        # ||T_hat||^2 = sum_{a,b,c} (sum_r U[a,r] V[b,r] W[c,r])^2
        # This is harder to compute quickly without forming T_hat.
        # However, we only need to check if Error is 0.
        # If Error is 0, then T_hat == T.
        # So we can just check if T_hat == T at the non-zero positions AND T_hat == 0 at zero positions.
        # But checking zero positions is expensive (10000 positions).
        
        # Alternative:
        # Check if T_hat matches T at non-zero positions.
        # AND check if the rank of the decomposition is sufficient to cover the zeros?
        # No, that's not enough.
        
        # Let's compute the full error but efficiently.
        # T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
        # This is fast enough (10k elements).
        T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T - T_hat) ** 2)

    # Local search with better error computation
    # We will use the fact that T is sparse to speed up the "inner product" part,
    # but we still need the norm of T_hat.
    # Actually, just using einsum is fine.
    
    # Strategy:
    # 1. Start with a random sparse initialization.
    # 2. Use coordinate descent / local search.
    # 3. If stuck, try a different rank or restart.
    
    start_time = time.time()
    time_limit = 110.0
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 60 upwards
    # Known lower bound is likely around 60-70.
    # Schoolbook is 100.
    # Let's try to find a solution with rank < 100.
    
    for rank in range(60, 101):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts
        max_restarts = 50
        found = False
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Sparse random init
            # Density ~ 0.2
            U = np.random.choice([-1, 0, 1], size=(dim_u, rank), p=[0.1, 0.8, 0.1])
            V = np.random.choice([-1, 0, 1], size=(dim_v, rank), p=[0.1, 0.8, 0.1])
            W = np.random.choice([-1, 0, 1], size=(dim_w, rank), p=[0.1, 0.8, 0.1])
            
            current_error = compute_error(U, V, W)
            if current_error == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Local Search
            # We will iterate until no improvement or time limit
            steps = 0
            while current_error > 0 and steps < 10000:
                if time.time() - start_time > time_limit:
                    break
                    
                steps += 1
                
                # Pick a random matrix and entry
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dim1, dim2 = dim_u, rank
                elif mat_idx == 1:
                    mat = V
                    dim1, dim2 = dim_v, rank
                else:
                    mat = W
                    dim1, dim2 = dim_w, rank
                    
                idx1 = np.random.randint(0, dim1)
                idx2 = np.random.randint(0, dim2)
                
                old_val = mat[idx1, idx2]
                
                # Try changing to -1, 0, 1
                best_val = old_val
                best_err = current_error
                
                for val in [-1, 0, 1]:
                    if val == old_val:
                        continue
                    mat[idx1, idx2] = val
                    err = compute_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_val = val
                    mat[idx1, idx2] = old_val
                
                if best_val != old_val:
                    mat[idx1, idx2] = best_val
                    current_error = best_err
                    if current_error == 0:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    # No improvement. To avoid getting stuck in local minima,
                    # we can accept a slightly worse move with some probability (Simulated Annealing)
                    # Or just break and restart.
                    # Given the time limit, breaking and restarting is safer.
                    break
        
        # If we found a solution with this rank, we could try to reduce it?
        # No, we are iterating ranks from low to high.
        # So the first one we find is the minimal rank for this search.
        # But we might miss it.
        # Let's record the best result found so far.
        # Actually, if we don't find 0 error, we can't use it.
        # We must return a valid decomposition.
        # So we fall back to schoolbook if we don't find a better one.
        
    return get_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
