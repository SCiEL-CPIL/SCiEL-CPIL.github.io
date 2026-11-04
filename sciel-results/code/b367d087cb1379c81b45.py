import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2x2 base decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def build_strassen_56():
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_block(base_rank, i_base, j_base, k_base):
            for r in range(7):
                cr = base_rank + r
                for (li, lj), val in strassen_U[r].items():
                    U[(i_base + li) * 4 + (j_base + lj)][cr] += val
                for (lj, lk), val in strassen_V[r].items():
                    V[(j_base + lj) * 8 + (k_base + lk)][cr] += val
                for (lk, li), val in strassen_W[r].items():
                    W[(k_base + lk) * 2 + (i_base + li)][cr] += val
        
        # 8 blocks: 2 (i-blocks) x 4 (j-blocks) x 1 (k-blocks of size 2)
        # Actually: i has 2 rows so 1 block of size 2, j has 4 cols so 2 blocks, k has 8 cols so 4 blocks
        # That's 1*2*4 = 8 blocks of 2x2x2
        blocks = []
        for ib in range(1):  # i-block: only 1 (since n=2)
            for jb in range(2):  # j-blocks: 2 (since m=4, block size 2)
                for kb in range(4):  # k-blocks: 4 (since p=8, block size 2)
                    blocks.append((ib*2, jb*2, kb*2))
        
        for idx, (i_base, j_base, k_base) in enumerate(blocks):
            add_block(idx * 7, i_base, j_base, k_base)
        
        return U, V, W, rank
    
    def compute_tensor(U_arr, V_arr, W_arr, rank_val):
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
        return tensor
    
    def is_valid(U_arr, V_arr, W_arr, rank_val, tol=1e-9):
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=tol)
    
    U_list, V_list, W_list, rank = build_strassen_56()
    U_np = np.array(U_list, dtype=float)
    V_np = np.array(V_list, dtype=float)
    W_np = np.array(W_list, dtype=float)
    
    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np, rank):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank
    
    # Strategy 1: Try to find linear dependencies among rank components
    # For each pair of columns, check if one is a linear combination of others
    # by examining the induced bilinear forms
    
    # More efficient: use the fact that if we have columns u_r, v_r, w_r,
    # and if there exist coefficients c_s such that for all (i,j,k):
    # sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # then column r is redundant.
    
    # This is equivalent to: the tensor slice T_r(i,j,k) = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # must be in the span of {T_s : s != r}
    
    # Build the tensor for each rank component
    def get_component_tensor(U_arr, V_arr, W_arr, r):
        """Get the tensor contribution from rank component r."""
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        return tensor
    
    # Check if any single component is in the span of others
    # Use least squares: for component r, solve min ||T_r - sum_{s!=r} c_s T_s||
    # This is a linear system in c_s
    
    # Precompute all component tensors
    comp_tensors = []
    for r in range(rank):
        comp_tensors.append(get_component_tensor(U_np, V_np, W_np, r))
    
    # Stack into matrix: each component is a vector of size n*m*p = 64
    tensor_matrix = np.array(comp_tensors).T  # shape (64, rank)
    
    # For each column r, check if it's in the span of the others
    # by computing the rank of tensor_matrix with and without column r
    removed_cols = []
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Check if column r is in span of others
        others = [c for c in range(rank) if c != r and c not in removed_cols]
        if not others:
            continue
        
        # Compute rank with and without column r
        M_without = tensor_matrix[:, others]
        M_with = tensor_matrix[:, [r] + others]
        
        rank_without = np.linalg.matrix_rank(M_without, tol=1e-8)
        rank_with = np.linalg.matrix_rank(M_with, tol=1e-8)
        
        if rank_with == rank_without:
            # Column r is in the span of others, so it's redundant
            removed_cols.append(r)
    
    if removed_cols:
        keep_cols = [c for c in range(rank) if c not in removed_cols]
        new_U = U_np[:, keep_cols]
        new_V = V_np[:, keep_cols]
        new_W = W_np[:, keep_cols]
        new_rank = len(keep_cols)
        
        if is_valid(new_U, new_V, new_W, new_rank):
            best_U = new_U
            best_V = new_V
            best_W = new_W
            best_rank = new_rank
    
    # Strategy 2: Try removing multiple columns at once using greedy approach
    # but with proper linear dependency check
    current_U = best_U.copy()
    current_V = best_V.copy()
    current_W = best_W.copy()
    current_rank = best_rank
    
    # Rebuild component tensors for current decomposition
    comp_tensors_cur = []
    for r in range(current_rank):
        comp_tensors_cur.append(get_component_tensor(current_U, current_V, current_W, r))
    tensor_matrix_cur = np.array(comp_tensors_cur).T
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            others = [c for c in range(current_rank) if c != r]
            if not others:
                break
            
            M_without = tensor_matrix_cur[:, others]
            M_with = tensor_matrix_cur[:, [r] + others]
            
            rank_without = np.linalg.matrix_rank(M_without, tol=1e-8)
            rank_with = np.linalg.matrix_rank(M_with, tol=1e-8)
            
            if rank_with == rank_without:
                # Remove column r
                keep = [c for c in range(current_rank) if c != r]
                current_U = current_U[:, keep]
                current_V = current_V[:, keep]
                current_W = current_W[:, keep]
                current_rank -= 1
                
                # Rebuild tensor matrix
                comp_tensors_cur = []
                for cr in range(current_rank):
                    comp_tensors_cur.append(get_component_tensor(current_U, current_V, current_W, cr))
                tensor_matrix_cur = np.array(comp_tensors_cur).T
                
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
    
    # Strategy 3: Try small rank reductions by modifying entries
    # For each pair of columns, try to see if we can merge them
    # by finding a linear combination that preserves the tensor
    
    if time.time() < deadline - 15:
        # Try: for each pair (r, s), see if we can express component r as
        # a linear combination of s and other components, then eliminate r
        # This is already covered by the linear dependency check above.
        
        # Try a different approach: perturb the decomposition slightly
        # and see if we can find a lower-rank equivalent
        import random
        random.seed(123)
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try to find a rank-55 or lower decomposition by
            # setting one entry to zero in each of U, V, W for a specific column
            # and checking if the tensor is still preserved
            
            test_U = best_U.copy()
            test_V = best_V.copy()
            test_W = best_W.copy()
            test_rank = best_rank
            
            # Pick a random column
            col = random.randint(0, test_rank - 1)
            
            # Try zeroing one entry in U for this column
            nonzero_U = np.nonzero(test_U[:, col])[0]
            if len(nonzero_U) > 0:
                row_u = nonzero_U[random.randint(0, len(nonzero_U)-1)]
                old_val = test_U[row_u, col]
                test_U[row_u, col] = 0
                
                if is_valid(test_U, test_V, test_W, test_rank, tol=1e-8):
                    # This entry can be zeroed
                    best_U = test_U.copy()
                    best_V = test_V.copy()
                    best_W = test_W.copy()
                else:
                    test_U[row_u, col] = old_val
            
            # Try zeroing one entry in V for this column
            nonzero_V = np.nonzero(test_V[:, col])[0]
            if len(nonzero_V) > 0:
                row_v = nonzero_V[random.randint(0, len(nonzero_V)-1)]
                old_val = test_V[row_v, col]
                test_V[row_v, col] = 0
                
                if is_valid(test_U, test_V, test_W, test_rank, tol=1e-8):
                    best_U = test_U.copy()
                    best_V = test_V.copy()
                    best_W = test_W.copy()
                else:
                    test_V[row_v, col] = old_val
            
            # Try zeroing one entry in W for this column
            nonzero_W = np.nonzero(test_W[:, col])[0]
            if len(nonzero_W) > 0:
                row_w = nonzero_W[random.randint(0, len(nonzero_W)-1)]
                old_val = test_W[row_w, col]
                test_W[row_w, col] = 0
                
                if is_valid(test_U, test_V, test_W, test_rank, tol=1e-8):
                    best_U = test_U.copy()
                    best_V = test_V.copy()
                    best_W = test_W.copy()
                else:
                    test_W[row_w, col] = old_val
    
    # Convert to lists and round to half-integers
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    for mat in [U_final, V_final, W_final]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-9:
                    mat[i][j] = rounded
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
