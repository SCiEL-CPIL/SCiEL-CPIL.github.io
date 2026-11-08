import numpy as np
import time
import random

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin

    n, m, p = 3, 4, 4
    
    # Target Tensor T[i, j, k] = 1 if i==k (and j is the intermediate index? No)
    # Matrix multiplication: C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = delta_{i, k}? No.
    # The tensor for matrix multiplication is usually defined as T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # The decomposition is T = sum_r U_r[i] V_r[j] W_r[k].
    # And T_{i, j, k} = 1 if j == k? No.
    # A[i, j] * B[j, k] contributes to C[i, k].
    # So the tensor component T_{i, j, k} is 1 if the indices align for multiplication?
    # Actually, the tensor T is such that <T, A \otimes B> = C.
    # T_{i, j, k} = 1 if j == k? No, that would be trace.
    # Standard definition: T_{i, j, k} = \delta_{j, k} is wrong.
    # T_{i, j, k} = 1 if j is the intermediate index.
    # Let's construct the target tensor explicitly.
    
    # A is n x m (3x4)
    # B is m x p (4x4)
    # C is n x p (3x4)
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices i (0..n-1), j (0..m-1), k (0..p-1).
    # T[i, j, k] = 1 if j == k? No.
    # The term A[i, j] B[j, k] is present.
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is T[i, j, k] = 1 if the indices match the multiplication structure.
    # Actually, T[i, j, k] = 1 if j is the contraction index.
    # In the standard tensor decomposition for MM:
    # T_{i, j, k} = 1 if j == k? No.
    # Let's look at the seed code.
    # U has shape (n*m, rank). Indices of U are (i, j) flattened.
    # V has shape (m*p, rank). Indices of V are (j, k) flattened.
    # W has shape (p*n, rank). Indices of W are (k, i) flattened?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i.
    
    # The tensor T is defined by:
    # sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = C[i, k] = sum_j A[i, j] B[j, k]
    # So the target value for the tensor component corresponding to (i, j, k) in the U,V,W product is:
    # If we view U, V, W as vectors for each r, the sum is a tensor.
    # The target tensor T has components T[i, j, k] such that:
    # T[i, j, k] = 1 if j == k? No.
    # The identity is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = sum_j A[i,j] B[j,k] is NOT right.
    # The decomposition is:
    # C[i,k] = sum_r (sum_j A[i,j] U[i,j,r]) * (sum_j B[j,k] V[j,k,r]) * W[k,i,r] ? No.
    
    # Standard Tensor Decomposition of MM:
    # T_{i, j, k} = \delta_{j, k} is the tensor for Trace?
    # For MM, the tensor is T_{i, j, k} = 1 if j is the shared index.
    # Actually, the tensor is T[i, j, k] = 1 if j == k is wrong.
    # The tensor T satisfies:
    # <T, A \otimes B> = C
    # T_{i, j, k} A_{i, j} B_{j, k} = C_{i, k}
    # This implies T_{i, j, k} = \delta_{j, k}? No.
    # If T_{i, j, k} = 1, then sum_{i,j,k} A_{i,j} B_{j,k} = scalar.
    # The correct tensor is T_{i, j, k} = 1 if j == k? No.
    # The tensor is T_{i, j, k} = \delta_{j, k} is for the trace of A B?
    
    # Let's derive from the seed code.
    # U[global_a_idx, col] where global_a_idx = i*m + j.
    # V[global_b_idx, col] where global_b_idx = j*p + k.
    # W[global_c_idx, col] where global_c_idx = k*n + i.
    
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This sum must equal C[i, k] = sum_j A[i, j] B[j, k].
    # This equality must hold for ALL A, B.
    # This implies that the tensor defined by the sum must be the "multiplication tensor".
    # The multiplication tensor T has components T_{i, j, k} such that:
    # T_{i, j, k} = 1 if j == k? No.
    # Let's check the indices.
    # The term A[i, j] B[j, k] appears in C[i, k].
    # So the coefficient of A[i, j] B[j, k] in the expression for C[i, k] is 1.
    # In the tensor decomposition, the coefficient is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # So we require:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i, j, k) is a valid triple?
    # No, A[i, j] and B[j, k] are independent variables.
    # The expression C[i, k] is linear in A and B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T has non-zero components only when the indices match the multiplication structure.
    # Specifically, T[i, j, k] = 1 if j == k? No.
    # The tensor T is defined by T[i, j, k] = 1 if j is the contraction index.
    # Actually, T[i, j, k] = 1 for all i, j, k? No.
    # Let's look at the term A[i, j] B[j, k].
    # The indices are i (row A), j (col A / row B), k (col B).
    # The output is C[i, k].
    # So the tensor component T[i, j, k] corresponds to the coefficient of A[i, j] B[j, k] in C[i, k].
    # This coefficient is 1.
    # So T[i, j, k] = 1 for all i, j, k?
    # No, C[i, k] is a specific entry.
    # The full tensor T is of size (n*m) x (m*p) x (p*n).
    # T[ (i,j), (j',k), (k',i') ] = 1 if i==i', j==j', k==k'?
    # Let's check:
    # sum_r U[i,j,r] V[j',k,r] W[k',i',r]
    # We want this to be 1 if i==i', j==j', k==k' (and the multiplication structure is respected).
    # Actually, the tensor T is the "identity" in the space of matrices?
    # T[ (i,j), (j,k), (k,i) ] = 1.
    # All other components are 0.
    
    # So, construct the target tensor T of shape (n*m, m*p, p*n).
    # T[ i*m+j, j*p+k, k*n+i ] = 1.
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T[idx_u, idx_v, idx_w] = 1.0
        return T

    T_target = get_target_tensor()
    
    # Initial decomposition from the seed (Rank 44)
    # We can reuse the seed code to get the initial U, V, W
    # But we want to try to reduce the rank.
    
    # Strategy: Try to find a decomposition of rank R < 44.
    # We will try R = 43, 42, etc.
    # We use a local search method.
    
    def try_find_decomposition(rank, max_iters=5000, tol=1e-9):
        # Initialize U, V, W with random small values
        # Or better, initialize with the first 'rank' terms of the known 44-term solution
        # and perturb the rest?
        # Let's try random initialization with scaling.
        
        best_res = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        # Use multiple restarts
        num_restarts = 10
        time_per_restart = (timeout - (time.time() - start_time)) / num_restarts
        
        for _ in range(num_restarts):
            if time.time() - start_time > timeout:
                break
                
            # Initialize
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Gradient Descent
            # Residual R = T_target - sum_r U[:,r] outer V[:,r] outer W[:,r]
            # We want to minimize ||R||^2
            
            lr = 0.01
            for it in range(max_iters):
                if time.time() - start_time > timeout:
                    return None
                
                # Compute Residual
                # Efficient computation:
                # Res = T_target
                # for r in range(rank):
                #     Res -= np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                
                # This is slow in pure Python loops.
                # Use numpy einsum for the sum.
                # Sum = np.einsum('ir,jr,kr->ijk', U, V, W)
                # This creates a large intermediate? No, it's direct.
                
                Sum = np.einsum('ir,jr,kr->ijk', U, V, W)
                Res = T_target - Sum
                res_norm = np.linalg.norm(Res)
                
                if res_norm < best_res:
                    best_res = res_norm
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if res_norm < tol:
                    break
                
                # Gradients
                # d/dU[i,r] = - sum_{j,k} Res[i,j,k] V[j,r] W[k,r]
                # dU = - np.einsum('ijk,jr,kr->ir', Res, V, W)
                dU = -np.einsum('ijk,jr,kr->ir', Res, V, W)
                dV = -np.einsum('ijk,ir,kr->jr', Res, U, W)
                dW = -np.einsum('ijk,ir,jr->kr', Res, U, V)
                
                # Update
                U += lr * dU
                V += lr * dV
                W += lr * dW
                
                # Optional: Scale back to prevent divergence
                # U /= np.linalg.norm(U)
                # V /= np.linalg.norm(V)
                # W /= np.linalg.norm(W)
                
        if best_res < 1e-5:
            # Snap to half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            # Verify
            Sum_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            if np.allclose(Sum_snap, T_target, atol=1e-9):
                return U_snap, V_snap, W_snap
                
        return None

    # Try rank 43
    result = try_find_decomposition(43)
    if result:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Try rank 42
    result = try_find_decomposition(42)
    if result:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to the known rank 44 solution from the seed
    # Reconstruct the seed solution
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3): 
            for c in range(2): 
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2): 
            for c in range(2): 
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3): 
            for c in range(2): 
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
