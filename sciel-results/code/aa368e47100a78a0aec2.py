"""Matrix multiplication 3x5x5 rank minimization."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 5]
TIME_LIMIT = 115.0  # safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] where:
    # T is a 3-tensor of shape (n, m, p) with T[i,j,k] = 1 if i,j,k form valid indices
    # Actually, the matrix multiplication tensor has entries:
    # T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j] * B[j,k])
    # Wait, let me reconsider. The tensor for matrix multiplication C = A * B where
    # A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 if the indices are valid, and it's a 
    # 3-way tensor of shape (n, m, p).
    
    # Actually, the standard formulation: the multiplication tensor is a 3-tensor
    # where T[i,j,k] = 1 for all valid i,j,k. This is just the all-ones tensor?
    # No. Let me think again.
    
    # The matrix multiplication tensor is defined such that:
    # <T, A ⊗ B> = C flattened
    # where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if the "output index" (i,k) matches, but that's not right either.
    
    # The correct formulation: T is a tensor of shape (n, m, p) where
    # T[i,j,k] = 1 for all i,j,k. Then the contraction is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] ... no that doesn't work.
    
    # Let me use the standard definition. The matrix multiplication tensor 
    # μ_{n,m,p} is a 3-tensor with indices (i,j,k) where i∈{0..n-1}, j∈{0..m-1}, k∈{0..p-1}
    # and μ[i,j,k] = 1 if and only if... actually all entries are 1? No.
    
    # The correct way: C = AB, C[i,k] = sum_j A[i,j]B[j,k]
    # We can write this as: sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} = sum_{i,j,k} (sum_j' A_{i,j'}B_{j',k}) A_{i,j} B_{j,k}
    # This isn't quite right. Let me think of it differently.
    
    # The tensor T has shape (n*p, m, m) or we can think of it as a linear map.
    # Actually, the standard approach: the multiplication tensor is a 3-tensor 
    # T of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k.
    # Then the bilinear form <T, A⊗B> gives sum_{i,j,k} A[i,j]*B[j,k] which is not C.
    
    # I think the correct formulation for the decomposition is:
    # We want U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # for all A (n x m) and B (m x p):
    #   C = A * B (n x p)
    #   and W^T @ (U^T @ A_flat) @ (V^T @ B_flat) ... 
    # Actually, the decomposition means:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... no
    # 
    # The correct decomposition: C = sum_c (U[:,c] outer V[:,c]) contracted with W[:,c]
    # More precisely: for each c, we have a rank-1 term.
    # U is (n*m) x rank, so U[i*m+j, c] is the coefficient for A[i,j] in term c
    # V is (m*p) x rank, so V[j*p+k, c] is the coefficient for B[j,k] in term c  
    # W is (p*n) x rank, so W[k*n+i, c] is the coefficient for C[k*n+i]... 
    # Wait, W[k*n+i, c] - the output index is k*n+i which corresponds to C[i,k]?
    # The problem says "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) -> position k*n+i, which means W corresponds to C[i,k]
    # with i being the row and k being the column of C.
    
    # So the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this needs to hold for all A, B, so:
    # sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # Hmm, that's not quite right either.
    
    # Actually the standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T has shape (n*m, m*p, p*n) where
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    # Then a rank-r decomposition means: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c_idx] = sum_r U[a,r]*V[b,r]*W[c_idx,r]
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 15
    num_B = m * p  # 25
    num_C = p * n  # 15
    
    # Build the tensor T as a 3D array of shape (num_A, num_B, num_C)
    T = np.zeros((num_A, num_B, num_C), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook rank is n*m*p = 75
    # We need to find minimal rank decomposition.
    
    # Strategy: Try to find a rank-r decomposition by solving the linear system.
    # For a given rank r, we need to find U (num_A x r), V (num_B x r), W (num_C x r)
    # such that T[a,b,c] = sum_{r_idx} U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
    
    # This is a non-convex problem. One approach: fix two matrices and solve for the third.
    # Another: use the fact that over finite fields, we can use Gaussian elimination.
    
    # Let's try a different approach: use the known lower bounds and try to find
    # decompositions of rank < 75.
    
    # Known: for 3x5x5, the rank is at most 75 (schoolbook). 
    # Can we do better? Let's try to find a decomposition of rank < 75.
    
    # Approach: Randomized search with local optimization.
    # Fix U and V, then for each c, the equation becomes:
    # T[:, :, c] = sum_r U[:, r] * V[:, r] * W[c, r]
    # This is linear in W if U and V are fixed!
    # Specifically: for each c, T[:, :, c] = sum_r W[c, r] * (U[:, r] ⊗ V[:, r])
    # So we have a linear system in W.
    
    # Similarly, we can fix U and W and solve for V, or fix V and W and solve for U.
    
    # Let's try an alternating minimization approach:
    # 1. Initialize U, V, W randomly
    # 2. Fix U, V, solve for W (linear least squares)
    # 3. Fix U, W, solve for V (linear least squares)  
    # 4. Fix V, W, solve for U (linear least squares)
    # 5. Repeat
    
    # But this finds approximate solutions. We need exact integer/half-integer solutions.
    
    # Alternative: Use the finite field approach. Work over F_2 or F_3.
    # If we can find a decomposition over F_q, we might be able to lift it.
    
    # Let me try a simpler approach first: just return the schoolbook decomposition
    # but see if we can find a smaller rank through search.
    
    # For now, let's try rank values from 75 down and see if we can find a valid decomposition.
    
    def verify_decomposition(U, V, W, T, tol=1e-9):
        """Check if U, V, W decompose T."""
        rank = U.shape[1]
        # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # Compute using einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    # Try to find a decomposition of rank < 75 using alternating least squares
    # over the reals, then round to integers.
    
    best_rank = n * m * p  # 75
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U_school = np.zeros((num_A, best_rank))
    V_school = np.zeros((num_B, best_rank))
    W_school = np.zeros((num_C, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to improve using alternating minimization
    for target_rank in range(74, 40, -1):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        found = False
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            # Initialize randomly
            U = np.random.randn(num_A, target_rank)
            V = np.random.randn(num_B, target_rank)
            W = np.random.randn(num_C, target_rank)
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() - start_time > TIME_LIMIT - 20:
                    break
                
                # Fix U, V, solve for W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each c: T[:,:,c] = sum_r W[c,r] (U[:,r] ⊗ V[:,r])
                # This is a linear system: for each c, we have num_A*num_B equations
                # in target_rank unknowns W[c, :]
                
                # Build the system: A_W @ W_flat = T_flat
                # where A_W[c, (a,b), r] = U[a,r] * V[b,r]
                A_W = np.einsum('ar,br->abcr', U, V)  # (num_A, num_B, num_C, target_rank)
                # Actually we need: for each c, sum_r A_W[a,b,c,r] * W[c,r] = T[a,b,c]
                # Reshape: A_W has shape (num_A, num_B, num_C, target_rank)
                # We want to solve: for each c, (A_W[:,:,c,:] reshaped) @ W[c,:] = T[:,:,c].flatten()
                
                # More efficiently: 
                # Let M be a matrix of shape (num_C * num_A * num_B, target_rank)
                # where M[(c, a, b), r] = U[a,r] * V[b,r]
                # And let y be of shape (num_C * num_A * num_B) where y[(c,a,b)] = T[a,b,c]
                
                M = np.einsum('ar,br,cr->abc r', U, V, np.eye(num_C))  # This is wrong
                # Let me do it differently
                M = np.zeros((num_C * num_A * num_B, target_rank))
                y = np.zeros(num_C * num_A * num_B)
                for c in range(num_C):
                    for a in range(num_A):
                        for b in range(num_B):
                            idx = c * num_A * num_B + a * num_B + b
                            for r in range(target_rank):
                                M[idx, r] = U[a, r] * V[b, r]
                            y[idx] = T[a, b, c]
                
                W_new, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
                W = W_new.reshape(num_C, target_rank)
                
                # Fix U, W, solve for V
                M = np.zeros((num_C * num_A * num_B, target_rank))
                y = np.zeros(num_C * num_A * num_B)
                for c in range(num_C):
                    for a in range(num_A):
                        for b in range(num_B):
                            idx = c * num_A * num_B + a * num_B + b
                            for r in range(target_rank):
                                M[idx, r] = U[a, r] * W[c, r]
                            y[idx] = T[a, b, c]
                
                V_new, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
                V = V_new.reshape(num_B, target_rank)
                
                # Fix V, W, solve for U
                M = np.zeros((num_C * num_A * num_B, target_rank))
                y = np.zeros(num_C * num_A * num_B)
                for c in range(num_C):
                    for a in range(num_A):
                        for b in range(num_B):
                            idx = c * num_A * num_B + a * num_B + b
                            for r in range(target_rank):
                                M[idx, r] = V[b, r] * W[c, r]
                            y[idx] = T[a, b, c]
                
                U_new, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
                U = U_new.reshape(num_A, target_rank)
            
            # Check if we have a good solution
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.max(np.abs(T_recon - T))
            
            if error < 1e-6:
                # Try to round to integers or half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_recon_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                error_round = np.max(np.abs(T_recon_round - T))
                
                if error_round < 1e-9:
                    best_rank = target_rank
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    found = True
                    break
                else:
                    # Try rounding to integers
                    U_round = np.round(U)
                    V_round = np.round(V)
                    W_round = np.round(W)
                    
                    T_recon_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    error_round = np.max(np.abs(T_recon_round - T))
                    
                    if error_round < 1e-9:
                        best_rank = target_rank
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
            elif error < 1e-3:
                # Close but not quite, try more iterations or different init
                pass
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
