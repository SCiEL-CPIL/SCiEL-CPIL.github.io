import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank to try
    # Start with a rank that is likely achievable
    # Schoolbook is 54. Winograd 23*2=46.
    # Let's try to find a decomposition of rank 46 first.
    # If we can find rank 40, even better.
    
    start_time = time.time()
    time_limit = 140
    
    # Define the target tensor T
    # T[i, j, k] = 1 for all i, j, k
    # Indices: i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U shape: (nm, R)
    # V shape: (mp, R)
    # W shape: (pn, R)
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Use dot product for speed
                    # But we need sum of U[r]*V[r]*W[r]
                    # This is element-wise product then sum
                    # Vectorized:
                    # prod = U[u_idx] * V[v_idx] * W[w_idx]
                    # val = np.sum(prod)
                    # But we are in a loop.
                    for r in range(len(U[0])):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    err += (val - 1.0) ** 2
        return err

    def compute_gradients(U, V, W):
        # Gradient of E w.r.t U[i, r]
        # dE/dU[i, r] = 2 * sum_{j,k} (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
        # Note: T[i,j,k] uses U[i,j,r], V[j,k,r], W[k,i,r]
        # The term U[i*m+j, r] appears in T[i, j, k] for all k.
        # So dE/dU[i*m+j, r] = 2 * sum_k (T[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
        
        R = len(U[0])
        dU = [[0.0] * R for _ in range(nm)]
        dV = [[0.0] * R for _ in range(mp)]
        dW = [[0.0] * R for _ in range(pn)]
        
        # Precompute T values and errors
        T_vals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    T_vals[(i, j, k)] = val
                    
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for r in range(R):
                    grad_u = 0.0
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        err = T_vals[(i, j, k)] - 1.0
                        grad_u += 2.0 * err * V[v_idx][r] * W[w_idx][r]
                    dU[u_idx][r] = grad_u
                    
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                for r in range(R):
                    grad_v = 0.0
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        err = T_vals[(i, j, k)] - 1.0
                        grad_v += 2.0 * err * U[u_idx][r] * W[w_idx][r]
                    dV[v_idx][r] = grad_v
                    
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                for r in range(R):
                    grad_w = 0.0
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        err = T_vals[(i, j, k)] - 1.0
                        grad_w += 2.0 * err * U[u_idx][r] * V[v_idx][r]
                    dW[w_idx][r] = grad_w
                    
        return dU, dV, dW

    def optimize(U, V, W, lr=0.01, max_iters=1000):
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
            err = compute_error(U, V, W)
            if err < 1e-8:
                break
            dU, dV, dW = compute_gradients(U, V, W)
            # Update
            for i in range(nm):
                for r in range(len(U[0])):
                    U[i][r] -= lr * dU[i][r]
            for i in range(mp):
                for r in range(len(V[0])):
                    V[i][r] -= lr * dV[i][r]
            for i in range(pn):
                for r in range(len(W[0])):
                    W[i][r] -= lr * dW[i][r]
        return U, V, W

    def round_to_half_ints(U, V, W):
        def round_row(row):
            return [round(x * 2) / 2 for x in row]
        return [round_row(row) for row in U], [round_row(row) for row in V], [round_row(row) for row in W]

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try different ranks
    best_result = None
    best_rank = 54
    
    # Fallback: Schoolbook
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = make_schoolbook()
    
    # Try to find a better decomposition
    for target_rank in [46, 44, 42, 40]:
        if time.time() - start_time > time_limit - 20:
            break
            
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Convert to list of lists for the optimizer
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        U_list, V_list, W_list = optimize(U_list, V_list, W_list, lr=0.05, max_iters=200)
        
        # Round to half-integers
        U_r, V_r, W_r = round_to_half_ints(U_list, V_list, W_list)
        
        if verify(U_r, V_r, W_r):
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = target_rank
            break # Found a valid one, try lower rank? Or break?
            # Let's break and try lower rank in next iteration of loop
            
    if best_result is None:
        best_result = fallback
    else:
        # Try to reduce rank further by zeroing columns
        # This is complex, so we skip for now
        pass
        
    return best_result
