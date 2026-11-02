import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit

    # Schoolbook decomposition (rank = n*m*p = 36)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try to find a rank-23 decomposition using BCLR-style construction
    # The 4x4x4 matrix multiplication tensor has rank 23 (BCLR 1998)
    # We need to find a rank-23 decomposition for the 3x3x4 sub-tensor
    
    # The matrix multiplication tensor T_{ijk} = 1 if (i,j,k) corresponds to
    # C[i,k] += A[i,j]*B[j,k]
    # For 3x3x4: i in [0,3), j in [0,3), k in [0,4)
    # T has entries T[i,j,k] = 1 for all valid (i,j,k)
    
    # We want U (9 x r), V (12 x r), W (12 x r) such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    # and = 0 for invalid combinations (but in this case all (i,j,k) are valid
    # for the 3x3x4 tensor)
    
    # Actually, the tensor for 3x3x4 matrix multiplication has exactly
    # n*m*p = 36 nonzero entries, one for each (i,j,k) triple.
    
    # Strategy: Use the BCLR 23-rank 4x4x4 decomposition and project to 3x3x4
    # The 4x4x4 BCLR decomposition gives us 23 rank-1 terms.
    # Each term has U_4 (4x4), V_4 (4x4), W_4 (4x4) factors.
    # We can take the top-left 3x3 submatrix of U_4, V_4, and adjust W_4.
    
    # However, this may not give exactly rank 23 for the 3x3x4 case.
    # Let's try a direct numerical approach with random initialization.
    
    if time.time() < deadline - 10:
        try:
            target_rank = 23
            rng = np.random.RandomState(42)
            
            # Initialize with small random values
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
            # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # for all i in [0,n), j in [0,m), k in [0,p)
            
            # This is a system of n*m*p = 36 equations in 3*9*23 + 3*12*23 + 3*12*23 = 
            # 621 + 828 + 828 = 2277 unknowns (with scaling ambiguity)
            
            # Use alternating least squares:
            # Fix V and W, solve for U: U[i*m+j, r] * (V*W)[i*m+j, r] = T[i*m+j]
            # where (V*W)[i*m+j, r] = sum_{j',k} V[j'*p+k, r] * W[k*n+i, r] * delta(j,j')
            # Actually this is more complex. Let me use a different approach.
            
            # The constraint is: for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This can be viewed as: for each r, we have a rank-1 tensor
            # U_r ⊗ V_r ⊗ W_r, and we want their sum to equal the target tensor.
            
            # Let's use gradient descent on the loss:
            # L = sum_{i,j,k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            
            # Compute gradient:
            # dL/dU[i*m+j, r] = 2 * (sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # summed over k (for fixed i,j)
            
            # Actually, let me just use a simple iterative approach.
            
            def compute_residuals(U, V, W):
                """Compute the residual tensor values."""
                residuals = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            residuals[i, j, k] = val - 1.0
                return residuals
            
            def compute_loss(U, V, W):
                res = compute_residuals(U, V, W)
                return np.sum(res ** 2)
            
            # Gradient descent with learning rate
            lr = 0.01
            best_loss = compute_loss(U, V, W)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Compute residuals
                res = compute_residuals(U, V, W)
                loss = np.sum(res ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res_ijk = res[i, j, k]
                            for r in range(target_rank):
                                dU[i * m + j, r] += 2 * res_ijk * V[j * p + k, r] * W[k * n + i, r]
                                dV[j * p + k, r] += 2 * res_ijk * U[i * m + j, r] * W[k * n + i, r]
                                dW[k * n + i, r] += 2 * res_ijk * U[i * m + j, r] * V[j * p + k, r]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if loss > best_loss * 1.1:
                    lr *= 0.9
                
                if iteration % 100 == 0:
                    # Check convergence
                    if loss < 1e-10:
                        break
            
            # Check if we found a good solution
            final_loss = compute_loss(best_U, best_V, best_W)
            if final_loss < 1e-10:
                # Verify and return
                # Round to nearest half-integer or integer if close
                best_U_rounded = np.round(best_U * 2) / 2
                best_V_rounded = np.round(best_V * 2) / 2
                best_W_rounded = np.round(best_W * 2) / 2
                
                # Check if rounded version still works
                rounded_loss = compute_loss(best_U_rounded, best_V_rounded, best_W_rounded)
                if rounded_loss < 1e-10:
                    best_result = {
                        "U": best_U_rounded.tolist(),
                        "V": best_V_rounded.tolist(),
                        "W": best_W_rounded.tolist()
                    }
                else:
                    best_result = {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
        
        except Exception as e:
            pass
    
    # Final check: return best result found
    return best_result
