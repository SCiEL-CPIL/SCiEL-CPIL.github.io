"""Program to find a rank-minimal tensor decomposition for 3x5x5 matrix multiplication."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 5]

# Deadline guard: 120s limit, use 115s for safety
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p  # 75
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Strategy: Use block decomposition
    # Split 3x5x5 into blocks:
    # A is 3x5, B is 5x5
    # Partition A into columns: A[:, 0:3] (3x3) and A[:, 3:5] (3x2)
    # Partition B into rows: B[0:3, :] (3x5) and B[3:5, :] (2x5)
    # And B into columns: B[:, 0:3] (5x3) and B[:, 3:5] (5x2)
    
    # The 3x3x3 block is the hardest - try to find rank 23 instead of 27
    # The other blocks can use schoolbook or small improvements
    
    # For 3x3x3, known best rank is 23
    # For 3x2x3, schoolbook gives 18, can we do better?
    # For 3x3x2, schoolbook gives 18, can we do better?
    # For 3x2x2, schoolbook gives 12
    
    # Let's try a numerical search for the 3x3x3 tensor at rank 23
    # The tensor T[i,j,k] = 1 if indices match the matrix multiplication pattern
    # For 3x3x3: T[i,j,k] contributes to C[i,k] from A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k in range(3)
    # Wait, that's not right. The tensor for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M[i,k,j] = delta(i,j) * delta(j,k) ... no.
    # Actually, the multiplication tensor has entries M[i,j,k] where:
    # It's a 3-tensor such that M[i,j,k] = 1 if the term A[i,j]*B[j,k] appears
    # So M is indexed by (i, j, k) and M[i,j,k] = 1 for all valid i,j,k
    # Wait, that would make it a rank n*m*p tensor trivially.
    
    # Let me reconsider. The matrix multiplication tensor T for (n,m,p) dimensions:
    # T is a tensor of shape (n*p, m*p) or similar... 
    # Actually, the standard formulation: we want to express C = A*B where
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor is T[i,k,j] = 1, meaning each term A[i,j]*B[j,k] contributes to C[i,k]
    # As a 3-tensor with modes (C_index, A_index, B_index):
    # T[(i,k), (i,j), (j,k)] = 1
    # Or equivalently, T[i,k,i,j,j,k] but that's 6 modes.
    
    # The factorization: C = U @ diag(A_vec) @ V^T ... 
    # The standard tensor decomposition:
    # T = sum_{c=0}^{rank-1} u_c ⊗ v_c ⊗ w_c
    # where T[i,k,j] = sum_c U[i,k,c] * V[i,j,c] * W[j,k,c] ... 
    # No, the interface says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And the decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes (i,j) for A, b indexes (j,k) for B, c indexes (i,k) for C
    # So T[(i,j),(j,k),(i,k)] = sum_r U[(i,j),r] * V[(j,k),r] * W[(i,k),r]
    # And we need T[(i,j),(j,k),(i,k)] = 1 for all valid indices
    # Wait, that's not quite right either. Let me re-read.
    
    # "W[k*n+i,r]" - so W is indexed by [k*n + i, r] where k is row of B / col of C
    # and i is row of A / row of C
    # U[i*m+j, r] where i is row of A, j is col of A / row of B
    # V[j*p+k, r] where j is row of B, k is col of B
    
    # The tensor equation: For each (i,j,k):
    # A[i,j] * B[j,k] contributes to C[i,k]
    # So the tensor T has T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3-way tensor of size (n*m) x (m*p) x (p*n) with entries that are 1
    # at positions corresponding to valid (i,j,k) tuples and 0 elsewhere.
    # Wait, no. Each (i,j,k) tuple gives ONE entry in the tensor that must equal 1.
    # But the tensor is indexed by (a, b, c) = (i*m+j, j*p+k, k*n+i)
    # Different (i,j,k) can give the same (a,b,c) if... let's check.
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For fixed (a,b,c), can there be multiple (i,j,k)?
    # From a: i = a // m, j = a % m
    # From b: j = b // p, k = b % p
    # From c: i = c % n, k = c // n
    # So i, j, k are uniquely determined by any one of a, b, c!
    # So the tensor has 1s at rank-many positions and 0s elsewhere.
    # The number of 1s is n*m*p = 75.
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And this must hold exactly.
    
    # This is a bilinear (actually trilinear) constraint.
    
    # For the 3x3x3 sub-problem, I'll use a known rank-23 decomposition.
    # Let me try to construct one or find it via search.
    
    # Actually, let me try a different approach: use the fact that we can
    # exploit structure. Let's do a random restart local search for the full problem
    # at rank 71 (saving 4 ranks from the 3x3x3 block going from 27 to 23).
    
    # But first, let me try to just improve the 3x3x3 block.
    # The 3x3x3 multiplication tensor at rank 23 is known to exist.
    
    # Let me implement a simple search for rank 23 for 3x3x3.
    # Represent U, V, W as 9x23 matrices (for 3x3x3, n=m=p=3, so n*m=9, m*p=9, p*n=9)
    
    def try_solve_333(rank_target, max_time=30.0):
        """Try to find rank r decomposition for 3x3x3 matrix multiplication."""
        t0 = time.time()
        n3 = m3 = p3 = 3
        size = n3 * m3  # 9
        
        # Build the target tensor indices
        # For each (i,j,k) in [0,3)^3:
        # U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :] should sum to 1
        # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        
        # This is a system of 27 equations (one per (i,j,k))
        # with 3 * 9 * rank_target = 27 * rank_target variables
        
        # Use random initialization and gradient descent / alternating least squares
        
        best_err = float('inf')
        best_solution = None
        
        for trial in range(20):
            if time.time() - t0 > max_time:
                break
            
            # Random initialization with small entries
            np.random.seed(42 + trial)
            U = np.random.randn(size, rank_target) * 0.5
            V = np.random.randn(size, rank_target) * 0.5
            W = np.random.randn(size, rank_target) * 0.5
            
            # ALS (Alternating Least Squares) style updates
            for it in range(200):
                if time.time() - t0 > max_time:
                    break
                
                # Update U: for each row a (which is i*m+j), solve least squares
                # sum_r U[a,r] * (V[:,r] * W[:,r] elementwise)[b,c] ... 
                # Actually, let's think of it differently.
                # For fixed V, W, we need for each (a, b, c) triple:
                # sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
                # where target[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
                # But wait, each (i,j,k) gives exactly one (a,b,c), and different
                # (i,j,k) give different (a,b,c). So the target tensor has 1s at
                # 27 positions and 0s elsewhere.
                
                # For updating U[a, :]: we need
                # sum_r U[a,r] * V[b,r] * W[c,r] = t[a,b,c] for all b,c
                # This is: U[a,:] @ (V^T * W^T elementwise) = t[a, :, :]
                # Let M[a, r] = U[a,r], N[r, b, c] = V[b,r] * W[c,r]
                # Then U[a,:] @ N = t[a,:,:]
                # So U[a,:] = t[a,:,:] @ N^+ (pseudo-inverse)
                
                # Build N: shape (rank, size, size) where N[r, b, c] = V[b,r] * W[c,r]
                N = V.T * W.T  # broadcasting: (rank, size) * (size, rank) -> (rank, size, size)? No.
                # V.T has shape (rank, size), W.T has shape (rank, size)
                # We want N[r, b, c] = V[b, r] * W[c, r]
                # = V.T[r, b] * W.T[r, c]
                N = V.T[:, :, None] * W.T[:, None, :]  # (rank, size, size)
                
                # For each a, solve: U[a,:] @ N_flat = t[a, :] where N_flat is N reshaped
                # N_flat has shape (rank, size*size)
                N_flat = N.reshape(rank_target, size * size)
                
                # t tensor: t[a, b, c] = 1 if valid, 0 otherwise
                # Build t as (size, size, size) -> flatten to (size, size*size)
                t_tensor = np.zeros((size, size, size))
                for i3 in range(n3):
                    for j3 in range(m3):
                        for k3 in range(p3):
                            a = i3 * m3 + j3
                            b = j3 * p3 + k3
                            c = k3 * n3 + i3
                            t_tensor[a, b, c] = 1.0
                t_flat = t_tensor.reshape(size, size * size)
                
                # Solve U @ N_flat^T = t_flat
                # U = t_flat @ pinv(N_flat^T)
                try:
                    U_new = t_flat @ np.linalg.pinv(N_flat.T)
                except:
                    U_new = U
                
                # Similarly for V and W
                # For V[b, :]: sum_r V[b,r] * U[a,r] * W[c,r] = t[a,b,c]
                # V[b,:] @ (U^T * W^T)[r, a, c] = t[:, b, :]
                M2 = U.T[:, :, None] * W.T[:, None, :]  # (rank, size, size)
                M2_flat = M2.reshape(rank_target, size * size)
                t_flat2 = t_tensor.transpose(1, 0, 2).reshape(size, size * size)  # t[b, a, c]
                try:
                    V_new = t_flat2 @ np.linalg.pinv(M2_flat.T)
                except:
                    V_new = V
                
                # For W[c, :]: sum_r W[c,r] * U[a,r] * V[b,r] = t[a,b,c]
                M3 = U.T[:, :, None] * V.T[:, None, :]  # (rank, size, size)
                M3_flat = M3.reshape(rank_target, size * size)
                t_flat3 = t_tensor.transpose(2, 0, 1).reshape(size, size * size)  # t[c, a, b]
                try:
                    W_new = t_flat3 @ np.linalg.pinv(M3_flat.T)
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check error
                err = 0.0
                for i3 in range(n3):
                    for j3 in range(m3):
                        for k3 in range(p3):
                            a = i3 * m3 + j3
                            b = j3 * p3 + k3
                            c = k3 * n3 + i3
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                err = np.sqrt(err)
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-8:
                    return U, V, W, err
        
        return best_solution[0], best_solution[1], best_solution[2], best_err
    
    # Try to solve 3x3x3 at rank 23
    if time.time() - start_time < DEADLINE - 60:
        try:
            U3, V3, W3, err3 = try_solve_333(23, max_time=min(60, DEADLINE - (time.time() - start_time) - 50))
            if err3 < 1e-6:
                print(f"Found 3x3x3 rank 23 solution with error {err3}")
                # Now build the full 3x5x5 solution using this
                # The full problem decomposes into blocks:
                # A is 3x5, B is 5x5
                # Split A into A1 (3x3) and A2 (3x2), B into B1 (3x5) and B2 (2x5)
                # C = A1*B1 + A2*B2
                # Block 1: 3x3x5 -> decompose as 3x3x3 + 3x3x2
                # Block 2: 3x2x5 -> decompose as 3x2x3 + 3x2x2
                
                # For 3x3x3: use the rank 23 solution
                # For 3x3x2: schoolbook gives 18, let's try rank 17
                # For 3x2x3: schoolbook gives 18, let's try rank 17  
                # For 3x2x2: schoolbook gives 12, let's try rank 11
                
                # Actually, let's just use schoolbook for the smaller blocks
                # to keep it simple and reliable.
                # Total rank: 23 + 18 + 18 + 12 = 71
                
                # Build the full solution
                total_rank = 23 + 18 + 18 + 12  # 71
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                col_offset = 0
                
                # Block 1: 3x3x3 (i in 0-2, j in 0-2, k in 0-2)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            a_idx = i * m + j  # i*5+j
                            b_idx = j * p + k  # j*5+k
                            c_idx = k * n + i  # k*3+i
                            # In the 3x3x3 subproblem, local indices:
                            # a_local = i*3+j, b_local = j*3+k, c_local = k*3+i
                            a_local = i * 3 + j
                            b_local = j * 3 + k
                            c_local = k * 3 + i
                            U_full[a_idx, col_offset:col_offset+23] = U3[a_local, :]
                            V_full[b_idx, col_offset:col_offset+23] = V3[b_local, :]
                            W_full[c_idx, col_offset:col_offset+23] = W3[c_local, :]
                col_offset += 23
                
                # Block 2: 3x3x2 (i in 0-2, j in 0-2, k in 3-4)
                # Schoolbook: rank 3*3*2 = 18
                for i in range(3):
                    for j in range(3):
                        for k in range(3, 5):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            U_full[a_idx, col_offset] = 1
                            V_full[b_idx, col_offset] = 1
                            W_full[c_idx, col_offset] = 1
                            col_offset += 1
                
                # Block 3: 3x2x3 (i in 0-2, j in 3-4, k in 0-2)
                # Schoolbook: rank 3*2*3 = 18
                for i in range(3):
                    for j in range(3, 5):
                        for k in range(3):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            U_full[a_idx, col_offset] = 1
                            V_full[b_idx, col_offset] = 1
                            W_full[c_idx, col_offset] = 1
                            col_offset += 1
                
                # Block 4: 3x2x2 (i in 0-2, j in 3-4, k in 3-4)
                # Schoolbook: rank 3*2*2 = 12
                for i in range(3):
                    for j in range(3, 5):
                        for k in range(3, 5):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            U_full[a_idx, col_offset] = 1
                            V_full[b_idx, col_offset] = 1
                            W_full[c_idx, col_offset] = 1
                            col_offset += 1
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    print(f"Full solution verified with max error {max_err}, rank {total_rank}")
                    return {
                        "U": U_full.tolist(),
                        "V": V_full.tolist(),
                        "W": W_full.tolist()
                    }
                else:
                    print(f"Verification failed with error {max_err}")
        except Exception as e:
            print(f"Error in 3x3x3 search: {e}")
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
