"""Find rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Tensor dimensions: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    dim_u = n * m  # 12
    dim_v = m * p  # 12
    dim_w = p * n  # 18
    
    # Build the target tensor T with T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # and i=i', j=j', k=k' (i.e., the j index matches)
    # More precisely: for A[i,j], B[j,k], C[i,k]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    
    # Build T as a 3D array
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p = 72)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try to find lower rank decompositions
    # For each rank r, try randomized initialization and Newton-like iterations
    for r in range(best_rank - 1, 30, -1):  # Try ranks from 71 down to 31
        if time.time() > deadline - 5:
            break
        success = False
        max_iter = 50
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Random initialization
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # Iteratively refine: for each rank-1 component, optimize
            # Use a simple alternating minimization
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                # Compute residuals
                # For each (a,b), we need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c
                # This is a linear system in W for fixed U,V: 
                # For each (a,b): W^T @ (U[:,a] * V[:,b]) = T[a,b,:]
                # Actually: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Let's define M_ab = U[:,a] * V[:,b] (vector of length r)
                # Then we need W^T @ M_ab = T[a,b,:] for all a,b
                # This is overdetermined. Use least squares.
                
                # Update W: for each c, W[:,c] should satisfy sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Collect all equations: for each (a,b,c): sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c]
                # Group by c: for fixed c, we have equations for all (a,b):
                # sum_r W[c,r] * U[a,r] * V[b,r] = T[a,b,c]
                # This is linear in W[c,:]. Let's form the system.
                
                # Alternative: use gradient descent on the full objective
                # F = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
                # Gradient w.r.t. W[c,r]: 2 * sum_{a,b} (pred[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
                
                # For efficiency, compute predictions and gradients
                # pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is expensive. Let's use a smarter approach.
                
                # Actually, let's just do a few iterations and check if we get close to exact
                # Compute current prediction
                # pred = np.einsum('ar,br,cr->abc', U, V, W)
                # This is O(r * dim_u * dim_v * dim_w) = O(r * 12 * 12 * 18)
                
                # Let's compute the error
                pred = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(pred - T)
                if err < 1e-10:
                    # Check exactness more carefully
                    if np.allclose(pred, T, atol=1e-12, rtol=0):
                        success = True
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
                # Simple gradient step
                # Compute gradient w.r.t. W
                # dW[c,r] = 2 * sum_{a,b} (pred[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
                diff = pred - T  # (dim_u, dim_v, dim_w)
                # dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                # = 2 * (U * V) contracted with diff
                # Let M[a,b,r] = U[a,r] * V[b,r]
                # dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * M[a,b,r]
                M = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (dim_u, dim_v, r)
                dW = 2 * np.einsum('abc,abr->cr', diff, M)  # (dim_w, r)
                # Similarly for U and V
                dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)  # (dim_u, r)
                dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)  # (dim_v, r)
                
                # Adaptive learning rate
                lr = 0.01 / (1 + it * 0.1)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if success:
                break
        
        if success:
            # Try even lower ranks
            continue
    
    # Convert to lists with integer/half-integer entries if possible
    # Check if entries are close to integers or half-integers
    def clean_matrix(M):
        M = np.round(M * 2) / 2  # Round to nearest half-integer
        return M.tolist()
    
    # Verify the solution
    U_final = np.array(best_U)
    V_final = np.array(best_V)
    W_final = np.array(best_W)
    pred_final = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    if not np.allclose(pred_final, T, atol=1e-10, rtol=0):
        # Fall back to schoolbook
        U_final, V_final, W_final = U_sb, V_sb, W_sb
        best_rank = n * m * p
    
    U_list = clean_matrix(U_final)
    V_list = clean_matrix(V_final)
    W_list = clean_matrix(W_final)
    
    # Final verification
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    pred_check = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    if not np.allclose(pred_check, T, atol=1e-10, rtol=0):
        # Fall back to schoolbook
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
