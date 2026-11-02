"""Matrix multiplication 6,6,6 - cyclic group Z_6 tensor decomposition approach."""
DIMENSIONS = [6, 6, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Approach: Use the structure of cyclic group Z_6 to build a rank-180 decomposition.
    # The idea: exploit periodicity modulo 6 in the indices.
    # 
    # For matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k], we can decompose using
    # the fact that 6 = 2*3, and use Chinese remainder-like structure.
    #
    # Actually, let me think about this more carefully. The standard Strassen approach
    # gives rank 189. Can we do better?
    #
    # Let me try a different structural approach: use the fact that 6 divides 12,
    # and construct a decomposition based on Z_12 group structure.
    #
    # Alternative: try to find a rank-180 decomposition by using a 3-level Strassen
    # or by exploiting the 2x3 block structure more cleverly.
    #
    # Let me first construct the Strassen rank-189 solution as baseline, then try
    # to reduce it further using linear algebra techniques.
    
    # First, construct the Strassen-based rank-189 decomposition
    rank_strassen = 7 * 27  # 189
    U = [[0.0] * rank_strassen for _ in range(n * m)]
    V = [[0.0] * rank_strassen for _ in range(m * p)]
    W = [[0.0] * rank_strassen for _ in range(p * n)]
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),              # M2
        ({0: 1}, {1: 1, 3: -1}),             # M3
        ({3: 1}, {2: 1, 0: -1}),             # M4
        ({0: 1, 1: 1}, {3: 1}),              # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7
    ]
    
    c_contrib = [
        {0: 1, 3: 1},   # M1: C00(+1), C11(+1)
        {2: 1, 3: -1},  # M2: C10(+1), C11(-1)
        {1: 1, 3: 1},   # M3: C01(+1), C11(+1)
        {0: -1, 2: 1},  # M4: C00(-1), C10(+1)
        {1: 1},         # M5: C01(+1)
        {0: 1},         # M6: C00(+1)
        {0: -1, 3: 1},  # M7: C00(-1), C11(+1)
    ]
    
    for s in range(7):
        a_coeffs = steps[s][0]
        b_coeffs = steps[s][1]
        c_coeffs = c_contrib[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    for a_block, a_sign in a_coeffs.items():
                        a_row = (a_block // 2) * 3 + i
                        a_col = (a_block % 2) * 3 + j
                        U[a_row * m + a_col][col] += float(a_sign)
                    
                    for b_block, b_sign in b_coeffs.items():
                        b_row = (b_block // 2) * 3 + j
                        b_col = (b_block % 2) * 3 + k
                        V[b_row * p + b_col][col] += float(b_sign)
                    
                    for c_block, c_sign in c_coeffs.items():
                        c_row = (c_block // 2) * 3 + i
                        c_col = (c_block % 2) * 3 + k
                        W[c_col * n + c_row][col] += float(c_sign)
    
    # Now try to reduce the rank using linear algebra
    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find dependencies among the rank-1 terms
    # Each column c defines a rank-1 term U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We want to find a subset of columns that still spans the tensor.
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if i==k else 0
    # In our decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Let's verify the current decomposition is correct
    # Build the tensor from the decomposition
    T_reconstructed = np.zeros((n, m, p), dtype=float)
    for c in range(rank_strassen):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    # Expected tensor
    T_expected = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T_expected[i, j, k] = 1.0
    
    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # Fallback to schoolbook if Strassen construction failed
        rank = n * m * p
        U_np = np.zeros((n*m, rank), dtype=float)
        V_np = np.zeros((m*p, rank), dtype=float)
        W_np = np.zeros((p*n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_np[i*m+j, column] = 1.0
                    V_np[j*p+k, column] = 1.0
                    W_np[k*n+i, column] = 1.0
                    column += 1
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    # Now try to reduce rank by finding linear dependencies
    # Represent each rank-1 term as a vector in the tensor space
    # The tensor space is R^(n*m*p) = R^216
    # Each term c gives a vector v_c in R^216 where v_c[i*m+j*p+k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # Build the matrix whose columns are the rank-1 vectors
    tensor_matrix = np.zeros((n*m*p, rank_strassen), dtype=float)
    for c in range(rank_strassen):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m*p) + j * p + k
                    tensor_matrix[idx, c] = U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    # The tensor T is the sum of all columns: T = tensor_matrix @ ones(rank_strassen)
    # We want to find a subset of columns that still sums to T.
    # This is equivalent to: find a vector w such that tensor_matrix @ w = T_vector
    # and minimize the number of non-zero entries in w (or find a sparse solution).
    
    T_vector = T_expected.reshape(-1)
    
    # Try to find a sparse representation using iterative methods
    # First, compute the rank of the tensor matrix
    try:
        rank_mm = np.linalg.matrix_rank(tensor_matrix)
    except:
        rank_mm = rank_strassen
    
    # If the columns are linearly dependent, we can potentially reduce rank
    # Use SVD to find the minimum number of terms needed
    # Actually, we need a different approach: find a subset of columns whose sum equals T_vector
    
    # Greedy approach: try to remove columns one by one and check if the sum still equals T
    # Start with all columns active
    active = np.ones(rank_strassen, dtype=bool)
    
    # Try removing each column and checking if the remaining still sum to T
    # This is O(rank^2 * tensor_size) which might be slow
    # Let's try a different approach: use the null space
    
    # Find null space of tensor_matrix
    try:
        _, s, vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        # Null space vectors are the rows of vh corresponding to zero singular values
        tol = 1e-10
        null_start = np.sum(s > tol)
        if null_start < rank_strassen:
            null_space = vh[null_start:]  # shape (rank - null_start, rank_strassen)
        else:
            null_space = np.array([])
    except:
        null_space = np.array([])
    
    # For each null space vector v, we can add v to the all-ones vector and get another valid solution
    # We want to find a solution w = ones + sum(a_i * null_vec_i) that is sparse
    
    # Try to find a sparse solution by optimization
    # This is hard in general, so let's try a simpler approach:
    # Check if we can set some entries to 0 by looking at the structure
    
    # Alternative: try to use the known rank bounds
    # Best known rank for 6x6x6 is at most 189 (Strassen)
    # Can we do better? The lower bound is at least 180 (from group algebra bounds)
    
    # Let's try a different structural decomposition
    # Use 2x3 blocks instead of 2x2 blocks
    
    # Actually, let me try to use the approach of splitting into 2x2 blocks of 3x3 matrices
    # but using a more efficient 3x3 multiplication.
    
    # The rank of 3x3 matrix multiplication: best known is 19? Let me check.
    # Actually, I'm not sure if rank 19 is achievable. The schoolbook is 27.
    # Let me try to use a rank-19 decomposition if I can find one.
    
    # For now, let's try to reduce the Strassen decomposition using linear algebra
    # more aggressively.
    
    # Try to find a subset of columns that works
    # Use a greedy set cover-like approach
    best_rank = rank_strassen
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Try to remove columns in a smart way
    # For each column, check if it's "needed" by seeing if the tensor sum changes
    # when we remove it
    
    # This is expensive, so let's limit the number of iterations
    max_iterations = 50
    current_sum = T_vector.copy()
    current_active = np.ones(rank_strassen, dtype=bool)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        # Try to find a column that can be removed
        # A column c can be removed if there exist coefficients for the other columns
        # that still sum to T_vector
        
        # Simple check: if column c is a linear combination of the others
        # This is expensive to check exactly, so let's use a heuristic
        
        # For each column, try to express it as a combination of the others
        # using least squares
        removed_any = False
        for c in range(rank_strassen):
            if not current_active[c]:
                continue
            
            # Check if column c is in the span of the other active columns
            other_indices = [i for i in range(rank_strassen) if i != c and current_active[i]]
            if len(other_indices) < 1:
                break
            
            sub_matrix = tensor_matrix[:, other_indices]
            
            # Solve sub_matrix @ x = tensor_matrix[:, c]
            try:
                x, residuals, rank_sub, sv = np.linalg.lstsq(sub_matrix, tensor_matrix[:, c], rcond=1e-10)
                reconstruction = sub_matrix @ x
                if np.allclose(reconstruction, tensor_matrix[:, c], atol=1e-8):
                    # Column c is redundant!
                    current_active[c] = False
                    removed_any = True
                    break
            except:
                pass
        
        if not removed_any:
            break
    
    if np.any(~current_active):
        new_rank = np.sum(current_active)
        if new_rank < best_rank:
            best_rank = new_rank
            keep_indices = np.where(current_active)[0]
            best_U = U_np[:, keep_indices]
            best_V = V_np[:, keep_indices]
            best_W = W_np[:, keep_indices]
    
    # Convert back to lists
    result_U = best_U.tolist()
    result_V = best_V.tolist()
    result_W = best_W.tolist()
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
