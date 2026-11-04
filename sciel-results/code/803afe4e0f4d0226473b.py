import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Prepare target tensor indices
    # T is non-zero at (i*m+j, j*p+k, k*n+i)
    indices_u = []
    indices_v = []
    indices_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_u.append(i * m + j)
                indices_v.append(j * p + k)
                indices_w.append(k * n + i)
    
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # Try ranks in descending order of likelihood of finding a solution?
    # Or ascending to find minimal?
    # If we find a solution for r, we can try r-1.
    # But finding a solution for small r is hard.
    # Let's try a few specific ranks that are likely to be valid.
    # Based on 49+25=74, let's try 75, 70, 65.
    
    candidate_ranks = [75, 70, 65, 60, 55]
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            # Simple gradient descent
            # Loss: sum over (u_idx, v_idx, w_idx) of (T - sum_c U[u,c]V[v,c]W[w,c])^2
            # T is 1 at these indices.
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute current values at target indices
                # val[c] = U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                # We want sum_c val[c] to be 1 for each target index.
                
                # This is a sum of products.
                # Let's compute the error.
                # err = 1 - sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c]
                
                # To compute gradients, we need the chain rule.
                # L = sum_t (1 - sum_c U[u_t, c] V[v_t, c] W[w_t, c])^2
                # dL/dU[u_t, c] = -2 * err_t * V[v_t, c] * W[w_t, c]
                
                # We can vectorize this.
                # U_sel = U[indices_u, :] shape (100, r)
                # V_sel = V[indices_v, :] shape (100, r)
                # W_sel = W[indices_w, :] shape (100, r)
                
                U_sel = U[indices_u, :]
                V_sel = V[indices_v, :]
                W_sel = W[indices_w, :]
                
                # Sum over c
                # prod = U_sel * V_sel * W_sel shape (100, r)
                prod = U_sel * V_sel * W_sel
                sums = np.sum(prod, axis=1) # shape (100,)
                errs = 1.0 - sums # shape (100,)
                
                # Check convergence
                if np.max(np.abs(errs)) < 1e-6:
                    found = True
                    break
                
                # Gradients
                # dL/dU_sel = -2 * errs[:, None] * V_sel * W_sel
                # We need to accumulate gradients for U, V, W.
                
                dU_sel = -2.0 * errs[:, None] * V_sel * W_sel
                dV_sel = -2.0 * errs[:, None] * U_sel * W_sel
                dW_sel = -2.0 * errs[:, None] * U_sel * V_sel
                
                # Accumulate into dU, dV, dW
                # Since indices might repeat? No, each (i,j,k) is unique.
                # So we can just assign back.
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                np.add.at(dU, indices_u, dU_sel)
                np.add.at(dV, indices_v, dV_sel)
                np.add.at(dW, indices_w, dW_sel)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            if found:
                # Convert to list of lists
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break # Try next (lower) rank
        
        if found:
            # We found a solution for this r.
            # We can continue to try lower ranks, or break if we want the first found?
            # The task asks for "rank-minimal". So we should keep trying lower.
            # But if we find 70, we should try 65.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
