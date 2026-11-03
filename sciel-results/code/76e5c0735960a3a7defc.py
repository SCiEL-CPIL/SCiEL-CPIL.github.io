import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if this term contributes to C[i,k], else 0
    # Actually the tensor is T[i,j,k] = delta (it's 1 for all i,j,k since it's the identity tensor)
    # Wait - the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k], so each triple (i,j,k) has coefficient 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_vec = U_arr[:, r]  # length n*m
        v_vec = V_arr[:, r]  # length m*p
        w_vec = W_arr[:, r]  # length p*n
        
        # Reshape
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        
        # Outer product contribution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    # The matrix multiplication tensor is all ones
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Strategy: Start with schoolbook rank-96, then use structured null-space projection
    # to find and remove linearly dependent rank terms
    
    # Step 1: Build schoolbook decomposition (rank 96)
    rank_schoolbook = n * m * p  # 96
    
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0] * rank_schoolbook
                v = [0.0] * rank_schoolbook
                w = [0.0] * rank_schoolbook
                
                idx = i * m * p + j * p + k
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    # Transpose: U should be (n*m, rank), V should be (m*p, rank), W should be (p*n, rank)
    U_arr = np.array(U_list, dtype=float).T  # (rank, n*m) -> transpose to (n*m, rank)
    V_arr = np.array(V_list, dtype=float).T  # (rank, m*p) -> transpose to (m*p, rank)
    W_arr = np.array(W_list, dtype=float).T  # (rank, p*n) -> transpose to (p*n, rank)
    
    # Step 2: Identify linear dependencies using structured null-space projection
    # The key insight: for matrix multiplication 3x4x8, we can exploit the fact that
    # some combinations of rank-1 terms are linearly dependent
    
    # Build the tensor as a matrix for SVD analysis
    # Tensor T has shape (n*m, m*p, p*n) = (12, 32, 24)
    # We can flatten to get a matrix and find its rank
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # The rank of the tensor is what we want to minimize
    # Use iterative approach: try to find low-rank decompositions
    
    # Step 3: Use Strassen-like block structure to get rank 88
    # Then try to reduce further
    
    # Build Strassen decomposition (rank 88)
    rank_strassen = 88
    U_s = np.zeros((n * m, rank_strassen))
    V_s = np.zeros((m * p, rank_strassen))
    W_s = np.zeros((p * n, rank_strassen))
    
    col = 0
    
    # Strassen factors for 2x2x2
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U_s[u_idx, col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V_s[v_idx, col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W_s[w_idx, col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U_s[u_idx, col] = 1
                V_s[v_idx, col] = 1
                W_s[w_idx, col] = 1
                col += 1
    
    # Verify Strassen decomposition
    if not _verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
        # Fallback to schoolbook if verification fails
        return {
            "U": U_arr.tolist(),
            "V": V_arr.tolist(),
            "W": W_arr.tolist()
        }
    
    # Step 4: Try to reduce rank further using null-space projection
    # Check if any columns are linearly dependent
    
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_strassen
    
    # Try to find linear dependencies in the Strassen decomposition
    # Use SVD to check for near-zero singular values
    
    # Flatten the tensor contributions
    def check_rank_reduction(U_arr, V_arr, W_arr, target_rank):
        """Try to reduce rank by checking if we can drop some columns."""
        current_rank = U_arr.shape[1]
        
        if current_rank <= target_rank:
            return U_arr, V_arr, W_arr, current_rank
        
        # Try to identify which columns can be removed
        # Check linear independence of columns in U, V, W simultaneously
        
        # Build a combined matrix to check dependencies
        # Each rank-1 term is defined by (u_col, v_col, w_col)
        # Two terms are dependent if one is a scalar multiple of the other in all three components
        
        # Use a different approach: try greedy elimination
        kept_indices = list(range(current_rank))
        
        # Try removing each column and checking if the decomposition still works
        for col_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing this column
            new_indices = [i for i in kept_indices if i != col_idx]
            if len(new_indices) < target_rank:
                continue
            
            U_test = U_arr[:, new_indices]
            V_test = V_arr[:, new_indices]
            W_test = W_arr[:, new_indices]
            
            # Quick check: verify the decomposition
            if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                kept_indices = new_indices
        
        return U_arr[:, kept_indices], V_arr[:, kept_indices], W_arr[:, kept_indices], len(kept_indices)
    
    # Try to reduce from 88
    U_reduced, V_reduced, W_reduced, new_rank = check_rank_reduction(U_s, V_s, W_s, 80)
    
    if new_rank < best_rank:
        best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
        best_rank = new_rank
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
