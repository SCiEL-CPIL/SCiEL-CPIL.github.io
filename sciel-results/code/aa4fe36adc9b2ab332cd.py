import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor
    # T[i, j, k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # A is n x m, indexed as A[i_a, i_b] with flat index i_a * m + i_b
    # B is m x p, indexed as B[j_a, j_b] with flat index j_a * p + j_b
    # C is n x p, but W uses trace convention: W[k*n + i, r] where k is row of C, i is col? 
    # Actually W[k*n+i, r] - let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # C[i_c, k_c] = sum_j A[i_c, j] * B[j, k_c]
    # W index: for C element at position (i_c, k_c), the W index is k_c * n + i_c
    
    # Build tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i_c in range(n):
        for j in range(m):
            for k_c in range(p):
                a_idx = i_c * m + j
                b_idx = j * p + k_c
                c_idx = k_c * n + i_c  # trace convention
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Now find rank decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each r, the slice T[:,:,r] (if we think of it as a 3-way) 
    # Actually, we can reshape: for a fixed r, U[:,r] * V[:,r]^T * W[:,r]^T gives a rank-1 tensor
    # T = sum_r outer(outer(U[:,r], V[:,r]), W[:,r])
    
    # Strategy: Use iterative rank reduction
    # Start with a known decomposition, try to reduce
    
    # First, let's get a baseline decomposition using a simple approach
    # Then try to reduce rank using linear algebra
    
    # Simple approach: use the structure to build an initial decomposition
    # Then use a solver to find if lower rank is possible
    
    # Let's use a greedy algorithm:
    # 1. Start with a valid decomposition (we'll construct one)
    # 2. For each rank r from lower to upper, check if rank-r decomposition exists
    
    # To check if rank-r decomposition exists, we can use the following:
    # Fix U and V, then for each c, T[:, :, c] should be in span of {U[:,r] * V[:,r]^T}
    # This is a linear system in W
    
    # Better approach: Use the fact that if we fix two matrices, the third is determined
    # Let's use an alternating least squares style approach
    
    # First, create a random or structured initial decomposition
    # Then iterate to reduce rank
    
    # Actually, let me use a direct approach:
    # The minimum rank is the tensor rank. We can estimate it using numerical methods.
    
    # Let's try ranks from a lower bound up to the known upper bound
    # Lower bound: max(n*m, m*p, p*n) / min(n*m, m*p, p*n) ... not quite
    # Actually, the rank must be at least max(n*m, m*p, p*n) / min(n, m, p) ... 
    # For 2x3x4: n*m=6, m*p=12, p*n=8
    # The rank must be at least ceil(6*12/8) = ceil(9) = 9? No, that's not right either.
    
    # Let's just try to find the minimum rank by checking if a decomposition exists
    
    def try_decompose(rank):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        # Initialize with random values
        np.random.seed(42)
        
        # We'll use an iterative approach
        # Fix U, solve for V and W, etc.
        
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Iterate
        for it in range(200):
            # Fix U and W, solve for V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each b: sum_c T[:,b,c] * W[c,:] = V[b,:] * (U^T * W^T)^T ... 
            # Let me think differently.
            
            # Reshape T to (n*m, m*p*p*n)
            T_flat = T.reshape(n * m, m * p * p * n)
            
            # For fixed U and W, the equation is linear in V
            # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Sum over c: sum_c T[a,b,c] * W[c,r'] = sum_r U[a,r] * V[b,r] * sum_c W[c,r']*W[c,r] ... no
            
            # Let me use a different approach: 
            # For fixed U, the equation becomes:
            # For each (b,c): T[:,b,c] = sum_r U[:,r] * (V[b,r] * W[c,r])
            # This is a linear system: T[:,b,c] = U * x_{b,c} where x_{b,c}[r] = V[b,r]*W[c,r]
            
            # Actually, let's use the standard ALS for CP decomposition
            # Fix V and W, solve for U:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each (b,c): T[:,b,c] = U * (V[b,:] * W[c,:])
            # So U = T_reshaped * (V*W)^+ where T_reshaped has shape (n*m, m*p*p*n)
            
            # Let's do proper ALS
            # Reshape T to (n*m, m*p*p*n)
            T_mat = T.reshape(n * m, m * p * p * n)
            
            for inner in range(50):
                # Fix V and W, solve for U
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each (b,c): T[:,b,c] = U * diag(V[b,:] * W[c,:])
                # Stack: T_mat = U * M where M[b,c,r] = V[b,r]*W[c,r]
                # M has shape (m*p*p*n, rank)
                M = (V.reshape(m * p, 1, rank) * W.reshape(1, p * n, rank)).reshape(m * p * p * n, rank)
                # T_mat = U @ M
                U_new, _, _, _ = np.linalg.lstsq(M, T_mat.T, rcond=None)
                U_new = U_new.T
                
                # Fix U and W, solve for V
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each (a,c): T[a,:,c] = V * diag(U[a,:] * W[c,:])
                # Reshape T to (m*p, n*m*p*n)
                T_mat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                M_v = (U.reshape(n * m, 1, rank) * W.reshape(1, p * n, rank)).reshape(n * m * p * n, rank)
                V_new, _, _, _ = np.linalg.lstsq(M_v, T_mat_v.T, rcond=None)
                V_new = V_new.T
                
                # Fix U and V, solve for W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each (a,b): T[a,b,:] = W * diag(U[a,:] * V[b,:])
                # Reshape T to (p*n, n*m*m*p)
                T_mat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                M_w = (U.reshape(n * m, 1, rank) * V.reshape(1, m * p, rank)).reshape(n * m * m * p, rank)
                W_new, _, _, _ = np.linalg.lstsq(M_w, T_mat_w.T, rcond=None)
                W_new = W_new.T
                
                U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            if err < 1e-10:
                return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: the rank must be at least the max of the "frontal" ranks
    # For our case, let's try from 12 up to 22
    
    for rank in range(12, 23):
        result = try_decompose(rank)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W]
            
            # Verify
            T_recon = np.einsum('ar,br,cr->abc', 
                                np.array(U_list), np.array(V_list), np.array(W_list))
            if np.allclose(T_recon, T, atol=1e-10):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}
