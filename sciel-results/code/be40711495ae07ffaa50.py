import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # The problem decomposes into 3 independent 2x3x3 multiplications
    # because p=9 = 3*3, and the k-index in B and C partitions into 3 sets of 3.
    
    # We try to find a rank r < 18 decomposition for 2x3x3.
    # If found, total rank = 3*r.
    # If not, total rank = 54 (trivial).
    
    def find_low_rank_2x3x3():
        # Target tensor for 2x3x3
        # U: 6xR, V: 9xR, W: 6xR
        # T[i*3+j, j*3+k, k*2+i] = 1
        n_s, m_s, p_s = 2, 3, 3
        dim_u = n_s * m_s
        dim_v = m_s * p_s
        dim_w = p_s * n_s
        
        target = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    target[i*m_s+j, j*p_s+k, k*n_s+i] = 1.0
        
        # Try rank 17
        r = 17
        best_solution = None
        best_error = np.inf
        
        # Use a few random restarts
        for restart in range(5):
            np.random.seed(42 + restart)
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                # Compute tensor
                # T = sum_t U[:,t] V[:,t] W[:,t]
                # This is slow to compute fully.
                # We only care about the entries that are 1 in target.
                # But we must ensure other entries are 0.
                # So we compute the full tensor difference.
                
                # Efficient computation of the tensor sum
                # T = np.einsum('it,jt,kt->ijk', U, V, W)
                T = np.einsum('it,jt,kt->ijk', U, V, W)
                diff = T - target
                error = np.sum(diff**2)
                
                if error < best_error:
                    best_error = error
                    best_solution = (U.copy(), V.copy(), W.copy())
                    
                if error < 1e-6:
                    break
                
                # Gradients
                # dT/dU = V W^T
                # grad_U = np.einsum('ijk,jt,kt->it', diff, V, W)
                # This is also expensive.
                # Let's use a simpler heuristic or just stop.
                # Given 120s limit, 5 restarts * 200 steps * einsum might be okay.
                
                grad_U = np.einsum('ijk,jt,kt->it', diff, V, W)
                grad_V = np.einsum('ijk,it,kt->jt', diff, U, W)
                grad_W = np.einsum('ijk,it,jt->kt', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
        # Check if best_solution is valid and close to integers
        if best_solution is not None and best_error < 1e-5:
            U, V, W = best_solution
            # Scale to integers?
            # Check if 2*U, 2*V, 2*W are close to integers
            U_int = np.round(2 * U) / 2
            V_int = np.round(2 * V) / 2
            W_int = np.round(2 * W) / 2
            
            # Verify exactness
            T_check = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
            if np.allclose(T_check, target, atol=1e-9):
                return U_int, V_int, W_int, r
        
        return None

    small_decomp = find_low_rank_2x3x3()
    
    if small_decomp:
        U_s, V_s, W_s, r = small_decomp
        total_rank = 3 * r
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Fill blocks
        for block in range(3):
            U[:, block*r:(block+1)*r] = U_s
            # V block indices: j*p_s + k
            # k is 0..2 for block 0, 3..5 for block 1, 6..8 for block 2
            # V_s indices are j*3 + k_local
            # So V[3*block : 3*(block+1), block*r:(block+1)*r] = V_s?
            # No, V index is j*3 + k.
            # For block 0, k=0,1,2. V indices 0..8.
            # For block 1, k=3,4,5. V indices 9..17.
            # V_s has indices 0..8 (j*3+k_local).
            # So V[3*block*3 : 3*(block+1)*3] ? No.
            # V row index is j*p + k. p=9.
            # Wait, in the 2x3x3 sub-problem, p_s=3.
            # V_s row index is j*3 + k_local.
            # In the full problem, V row index is j*9 + k.
            # For block b, k = 3*b + k_local.
            # So V row index = j*9 + 3*b + k_local.
            # This is not a contiguous block in V for fixed b if we iterate j.
            # We need to map V_s rows to V rows.
            
            for j in range(3):
                for k_local in range(3):
                    v_s_idx = j * 3 + k_local
                    v_full_idx = j * 9 + (3 * block + k_local)
                    V[v_full_idx, block*r:(block+1)*r] = V_s[v_s_idx]
                    
            # W block indices: k*n + i
            # W_s row index is k_local * 2 + i.
            # W full row index is k * 2 + i = (3*block + k_local)*2 + i.
            for k_local in range(3):
                for i in range(2):
                    w_s_idx = k_local * 2 + i
                    w_full_idx = (3 * block + k_local) * 2 + i
                    W[w_full_idx, block*r:(block+1)*r] = W_s[w_s_idx]
                    
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to trivial rank 54
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
