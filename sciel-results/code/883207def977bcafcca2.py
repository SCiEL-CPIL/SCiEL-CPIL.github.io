import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = 2, 4, 4
    num_U = n * m  # 8
    num_V = m * p  # 16
    num_W = p * n  # 8
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Tensor flattening: for each rank-1 component, we have u_i * v_j * w_k
    # The tensor T can be written as sum over r of u_r[i] * v_r[j] * w_r[k]
    
    # Start with the known rank-28 Strassen-based solution
    best_rank = 28
    best_U = None
    best_V = None
    best_W = None
    
    # Generate the rank-28 solution using block Strassen
    def generate_strassen_solution():
        rank = 28
        U = [[0] * rank for _ in range(num_U)]
        V = [[0] * rank for _ in range(num_V)]
        W = [[0] * rank for _ in range(num_W)]
        
        strassen_steps = [
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
        ]
        
        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s
                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = l_row
                    global_j = a_col_start + l_col
                    global_idx = global_i * m + global_j
                    U[global_idx][col] += coeff
                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = b_row_start + l_row
                    global_j = b_col_start + l_col
                    global_idx = global_i * p + global_j
                    V[global_idx][col] += coeff
                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_k = c_col_start + l_col
                    global_i = l_row
                    global_idx = global_k * n + global_i
                    W[global_idx][col] += coeff
        
        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)
        
        return U, V, W
    
    U, V, W = generate_strassen_solution()
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to find lower rank using iterative rank reduction
    # We'll attempt to remove one component at a time and see if we can still represent the tensor
    # by adjusting the remaining components
    
    def verify_decomposition(U_arr, V_arr, W_arr, rank):
        """Check if the decomposition exactly represents the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank reduction from 28 down to some minimum
    current_rank = best_rank
    current_U = np.array(best_U, dtype=float)
    current_V = np.array(best_V, dtype=float)
    current_W = np.array(best_W, dtype=float)
    
    # Attempt to reduce rank by trying to eliminate zero or near-zero columns
    for target_rank in range(best_rank - 1, max(20, best_rank - 8), -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a subset of columns that can span the tensor
        # Use a greedy approach: try removing each column and see if we can adjust others
        success = False
        for remove_col in range(current_rank):
            if time.time() > deadline - 10:
                break
                
            # Create new matrices without this column
            U_new = np.delete(current_U, remove_col, axis=1)
            V_new = np.delete(current_V, remove_col, axis=1)
            W_new = np.delete(current_W, remove_col, axis=1)
            
            # Check if the remaining columns can still represent the tensor
            # This is a linear system: for each (i,j,k), sum over remaining r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # This is nonlinear, so we need a different approach
            
            # Instead, check if the column we're removing is redundant
            # i.e., if it can be expressed as a linear combination of others in a way that preserves the tensor
            
            # For now, just check if removing it breaks the decomposition
            if verify_decomposition(U_new, V_new, W_new, current_rank - 1):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                success = True
                break
        
        if not success:
            break
    
    # Convert back to lists with integer/half-integer values
    def round_to_half_int(arr):
        """Round values to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    final_U = round_to_half_int(current_U)
    final_V = round_to_half_int(current_V)
    final_W = round_to_half_int(current_W)
    
    # Verify the final result
    if verify_decomposition(final_U, final_V, final_W, current_rank):
        best_rank = current_rank
        best_U = [list(map(float, row)) for row in final_U]
        best_V = [list(map(float, row)) for row in final_V]
        best_W = [list(map(float, row)) for row in final_W]
    
    return {"U": best_U, "V": best_V, "W": best_W}
