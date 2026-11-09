"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    n_blocks = n // 2
    m_blocks = m // 2
    p_blocks = p // 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7
    
    import numpy as np
    
    def verify_decomp_fast(U_list, V_list, W_list):
        """Check if decomposition is valid using vectorized computation."""
        Ua = np.array(U_list, dtype=float)
        Va = np.array(V_list, dtype=float)
        Wa = np.array(W_list, dtype=float)
        rank = Ua.shape[1]
        
        # For each rank-1 term r, compute the outer product contribution
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            u_col = Ua[:, r]  # shape (n*m,)
            v_col = Va[:, r]  # shape (m*p,)
            w_col = Wa[:, r]  # shape (p*n,)
            
            # Reshape and compute
            u_2d = u_col.reshape(n, m)
            v_2d = v_col.reshape(m, p)
            w_2d = w_col.reshape(p, n)
            
            # T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            # This is a rank-1 tensor in the (i,j,k) space with the specific indexing
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
        
        return np.allclose(T_decomp, 1.0, atol=1e-9)
    
    # Verify current decomposition
    if not verify_decomp_fast(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Attempt to reduce rank by finding linearly dependent rank-1 components
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = total_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition using random linear combinations
    # and checking if the tensor is preserved
    import random
    random.seed(42)
    
    # Strategy: try random projections to reduce dimension
    # If we can find a matrix P of shape (current_rank, target_rank) such that
    # U' = U @ P, V' = V @ inv(P), W' = W @ inv(P) preserves the tensor
    # This is equivalent to finding a basis change in the rank space
    
    # Simpler approach: try to identify if any subset of columns can be removed
    # by checking if the corresponding rank-1 tensors are linearly dependent
    
    # Build the tensor T from the decomposition
    T_target = np.ones((n, m, p))
    
    # For each rank-1 component, compute its contribution to T
    # comp_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to see if any comp_r can be expressed as a linear combination of others
    
    # This is a hard problem. Let's try a heuristic: 
    # Randomly try to reduce by combining two components into one
    # and checking if the tensor is preserved (approximately)
    
    def try_reduce_rank(U_cur, V_cur, W_cur, max_iter=1000):
        """Try to reduce rank by merging components."""
        Uc = [row[:] for row in U_cur]
        Vc = [row[:] for row in V_cur]
        Wc = [row[:] for row in W_cur]
        rank = len(Uc[0])
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if rank <= 1:
                break
            
            # Pick two random components to try to merge
            r1, r2 = random.sample(range(rank), 2)
            
            # Try to see if comp_r2 can be absorbed into comp_r1
            # or if we can find a new component that replaces both
            # Simple test: check if removing r2 still gives valid decomposition
            # (this only works if comp_r2 is zero, which it won't be)
            
            # Better: try a random linear combination
            # New component r_new = a*comp_r1 + b*comp_r2
            # Remove r1 and r2, add r_new
            # Check if tensor is preserved
            
            # For efficiency, just try removing one component at a time
            # and see if we can compensate by scaling others
            # This is getting complex. Let's just return what we have.
            break
        
        return Uc, Vc, Wc
    
    # Given the complexity, let's try a more direct approach:
    # Use the structure of the problem. Since p=6=3*2, we have 3 blocks of size 2 in the output.
    # Each 4x4x2 sub-problem might have a lower rank than 2*2*7=28.
    
    # Let's try to construct a rank-10 decomposition for 4x4x2
    # by solving a system of equations
    
    def find_4x4x2_decomposition():
        """Find a rank-10 decomposition for 4x4x2 matrix multiplication."""
        # T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,2)
        # We need: sum_{r=1}^{10} U[i*4+j, r] * V[j*2+k, r] * W[k*4+i, r] = 1
        
        # Let's try to construct this using a structured approach
        # Split into 4x2x2 and 4x2x2 sub-problems? No, that doesn't help.
        
        # Try: use 10 rank-1 terms with specific structure
        # Based on known results, rank(4,4,2) = 10
        
        # Construct using the following pattern:
        # For k=0 and k=1, we need separate handling
        
        # Let's use a basis approach. The space of 4x4x2 tensors has dimension 4*4*2=32.
        # We need 10 rank-1 tensors that span the "all-ones" tensor.
        
        # I'll use a constructive method:
        # Define 10 vectors u_r in R^16, v_r in R^8, w_r in R^8
        # such that sum_r u_r[i*4+j] * v_r[j*2+k] * w_r[k*4+i] = 1
        
        # This is a system of 32 equations in 10*(16+8+8)=320 unknowns
        # We can solve it using least squares or by construction
        
        # Let me try a specific construction:
        # Use the fact that for 4x4x2, we can write:
        # C[i,0] = sum_j A[i,j]*B[j,0]
        # C[i,1] = sum_j A[i,j]*B[j,1]
        
        # Try using 10 products of the form:
        # (linear combo of A's) * (linear combo of B's)
        
        # Based on literature, one known rank-10 decomposition uses:
        # s1 = A[0,0]*B[0,0]
        # s2 = A[0,1]*B[1,0]
        # s3 = A[0,2]*B[2,0]
        # s4 = A[0,3]*B[3,0]
        # s5 = A[1,0]*B[0,1]
        # s6 = A[1,1]*B[1,1]
        # s7 = A[1,2]*B[2,1]
        # s8 = A[1,3]*B[3,1]
        # s9 = (A[2,0]+A[3,0])*B[0,0] + (A[2,1]+A[3,1])*B[1,0] + ... 
        # This is getting complicated. Let me just use a numerical solver.
        
        n_sub, m_sub, p_sub = 4, 4, 2
        target_rank = 10
        
        # Set up the linear system
        # Variables: U (16*10), V (8*10), W (8*10) = 320 variables
        # Equations: 4*4*2 = 32 equations
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear. Let's use a different approach.
        # Initialize randomly and use iterative refinement.
        
        np.random.seed(42)
        U_try = np.random.randn(n_sub * m_sub, target_rank) * 0.1
        V_try = np.random.randn(m_sub * p_sub, target_rank) * 0.1
        W_try = np.random.randn(p_sub * n_sub, target_rank) * 0.1
        
        # Iteratively refine
        for iteration in range(1000):
            if time.time() > deadline - 30:
                break
            
            # Compute current tensor
            T_cur = np.zeros((n_sub, m_sub, p_sub))
            for r in range(target_rank):
                u_col = U_try[:, r]
                v_col = V_try[:, r]
                w_col = W_try[:, r]
                u_2d = u_col.reshape(n_sub, m_sub)
                v_2d = v_col.reshape(m_sub, p_sub)
                w_2d = w_col.reshape(p_sub, n_sub)
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            T_cur[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
            
            error = np.max(np.abs(T_cur - 1.0))
            if error < 1e-9:
                break
            
            # Gradient descent step (simplified)
            # For each r, adjust U[:,r], V[:,r], W[:,r] to reduce error
            for r in range(target_rank):
                u_col = U_try[:, r].copy()
                v_col = V_try[:, r].copy()
                w_col = W_try[:, r].copy()
                
                # Compute gradient with respect to u_col
                # dT/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] summed over k
                grad_u = np.zeros(n_sub * m_sub)
                grad_v = np.zeros(m_sub * p_sub)
                grad_w = np.zeros(p_sub * n_sub)
                
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            err = T_cur[i, j, k] - 1.0
                            # dT[i,j,k]/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
                            grad_u[i * m_sub + j] += err * v_col[j * p_sub + k] * w_col[k * n_sub + i]
                            grad_v[j * p_sub + k] += err * u_col[i * m_sub + j] * w_col[k * n_sub + i]
                            grad_w[k * n_sub + i] += err * u_col[i * m_sub + j] * v_col[j * p_sub + k]
                
                lr = 0.01
                U_try[:, r] -= lr * grad_u
                V_try[:, r] -= lr * grad_v
                W_try[:, r] -= lr * grad_w
        
        # Check if we got a valid solution
        T_final = np.zeros((n_sub, m_sub, p_sub))
        for r in range(target_rank):
            u_col = U_try[:, r]
            v_col = V_try[:, r]
            w_col = W_try[:, r]
            u_2d = u_col.reshape(n_sub, m_sub)
            v_2d = v_col.reshape(m_sub, p_sub)
            w_2d = w_col.reshape(p_sub, n_sub)
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        T_final[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
        
        if np.max(np.abs(T_final - 1.0)) < 1e-6:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U_try.T]
            V_list = [[float(x) for x in row] for row in V_try.T]
            W_list = [[float(x) for x in row] for row in W_try.T]
            return U_list, V_list, W_list
        else:
            return None
    
    # Try to find the 4x4x2 decomposition
    decomp_442 = find_4x4x2_decomposition()
    
    if decomp_442 is not None:
        U_442, V_442, W_442 = decomp_442
        # Combine 3 blocks of 4x4x2 to get 4x4x6
        total_rank_new = 3 * len(U_442[0])
        
        U_new = [[0] * total_rank_new for _ in range(n * m)]
        V_new = [[0] * total_rank_new for _ in range(m * p)]
        W_new = [[0] * total_rank_new for _ in range(p * n)]
        
        for bk in range(3):
            col_offset = bk * len(U_442[0])
            for i in range(4):
                for j in range(4):
                    for r in range(len(U_442[0])):
                        U_new[i * m + j][col_offset + r] = U_442[i * 4 + j][r]
            for j in range(4):
                for k in range(2):
                    for r in range(len(U_442[0])):
                        V_new[j * p + (bk * 2 + k)][col_offset + r] = V_442[j * 2 + k][r]
            for k in range(2):
                for i in range(4):
                    for r in range(len(U_442[0])):
                        W_new[(bk * 2 + k) * n + i][col_offset + r] = W_442[k * 4 + i][r]
        
        # Verify
        if verify_decomp_fast(U_new, V_new, W_new):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback to the rank-84 solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
