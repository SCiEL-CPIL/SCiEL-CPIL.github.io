import numpy as np
import time

n, m, p = 2, 5, 6
TOTAL_TERMS = n * m * p  # 60

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_decomposition():
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        if col != 54:
            return None, None, None
        
        return U, V, W

    def try_reduce_rank(U, V, W, target_rank):
        """Try to reduce rank by eliminating columns."""
        R = U.shape[1]
        if R <= target_rank:
            return U, V, W
        
        # Build the tensor constraint matrix
        # For each (i,j,k), we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is trilinear. To linearize, fix two matrices and solve for the third.
        
        # Strategy: try to eliminate one column at a time
        # For column r, we want to find alpha_s (s != r) such that:
        # U[:,r] * V[:,r] * W[:,r] = sum_{s != r} alpha_s * U[:,s] * V[:,s] * W[:,s]
        # in the sense that the resulting tensor contributions match.
        
        # Build linear system: for each (i,j,k):
        # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = sum_{s!=r} alpha_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
        
        for r_to_eliminate in range(R):
            if time.time() > deadline - 10:
                break
            
            # Build A matrix and b vector
            A_rows = []
            b_vals = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        lhs = U[u_idx, r_to_eliminate] * V[v_idx, r_to_eliminate] * W[w_idx, r_to_eliminate]
                        
                        row = []
                        for s in range(R):
                            if s == r_to_eliminate:
                                continue
                            row.append(U[u_idx, s] * V[v_idx, s] * W[w_idx, s])
                        A_rows.append(row)
                        b_vals.append(lhs)
            
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Solve least squares
            try:
                alpha, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is good
            if np.max(np.abs(A @ alpha - b)) < 1e-8:
                # We can eliminate column r_to_eliminate
                # Create new matrices without this column
                keep = [s for s in range(R) if s != r_to_eliminate]
                
                U_new = np.zeros((n * m, R - 1))
                V_new = np.zeros((m * p, R - 1))
                W_new = np.zeros((p * n, R - 1))
                
                for new_idx, s in enumerate(keep):
                    U_new[:, new_idx] = U[:, s]
                    V_new[:, new_idx] = V[:, s]
                    W_new[:, new_idx] = W[:, s]
                
                # Add alpha contributions to the kept columns
                # Wait, this doesn't work directly because alpha_s multiplies the ENTIRE rank-1 tensor
                # U[:,s] * V[:,s] * W[:,s], not just W.
                # 
                # Actually, the decomposition is:
                # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
                # If we eliminate r, we need:
                # sum_{s!=r} (U[:,s] ⊗ V[:,s] ⊗ W[:,s] + alpha_s * ???)
                # 
                # This approach doesn't work because the alpha_s would need to modify U, V, W simultaneously.
                # Let me think differently.
                
                # Actually, the correct approach: if column r can be written as a linear combination
                # in the tensor space, we can't simply add alpha_s to the other columns because
                # the tensor product structure means we'd need to modify all three matrices.
                
                # Better approach: use the fact that if we fix U and V, the W matrix must satisfy
                # linear constraints. Let me try a different elimination strategy.
                
                pass
        
        return U, V, W

    # Build initial Strassen decomposition (rank 54)
    U_str, V_str, W_str = build_strassen_decomposition()
    
    if U_str is not None:
        if check_decomposition(U_str, V_str, W_str):
            return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
