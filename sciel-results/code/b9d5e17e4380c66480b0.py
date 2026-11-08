import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    def verify_333_numpy(U, V, W):
        """
        Verify 3x3x3 decomposition using numpy.
        U, V, W are lists of lists (9xR).
        Returns True if exact, False otherwise.
        """
        R = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to (3, 3, R)
        U_3d = U_arr.reshape(3, 3, R)
        V_3d = V_arr.reshape(3, 3, R)
        W_3d = W_arr.reshape(3, 3, R)
        
        # Compute tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # We need to sum over r.
        # T[i, j, k] = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
        # Note: W is indexed by (k, i, r) in the product?
        # In the problem description: W[k*n+i, r]. So W index is (k, i).
        # My W_3d is reshaped from (9, R) where row index is k*3+i.
        # So W_3d[k, i, r] is correct.
        
        T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U_3d, V_3d, W_3d)
        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        
        # Check if entries are integer or half-integer
        # The problem asks for integer/half-integer entries.
        # We verify exactness with float, but we should ensure the input was "clean".
        # Since we generate them, they should be.
        return True

    def search_333_decomp(target_rank, time_limit):
        """
        Search for a 3x3x3 decomposition of rank <= target_rank.
        Returns (U, V, W, R) or None.
        """
        start_time = time.time()
        R = target_rank
        
        # Initialize with random integers in [-2, 2]
        # We will use a local search approach.
        
        for trial in range(200):
            if time.time() - start_time > time_limit:
                break
            
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(9)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(9)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(9)]
            
            # Convert to numpy for faster evaluation
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            U_3d = U_arr.reshape(3, 3, R)
            V_3d = V_arr.reshape(3, 3, R)
            W_3d = W_arr.reshape(3, 3, R)
            
            # Compute initial error
            T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U_3d, V_3d, W_3d)
            error = np.sum((T - 1.0) ** 2)
            
            best_error = error
            best_U, best_V, best_W = U, V, W
            
            # Local search
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                
                if error < 1e-9:
                    # Check if entries are valid (they are, by construction)
                    return U, V, W, R
                
                # Pick a random variable to perturb
                # There are 3 * 9 * R variables
                var_idx = random.randint(0, 2 * 9 * R - 1)
                matrix_id = var_idx // (9 * R) # 0: U, 1: V, 2: W
                elem_idx = var_idx % (9 * R)
                row = elem_idx // R
                col = elem_idx % R
                
                # Get current value
                if matrix_id == 0:
                    old_val = U_arr[row, col]
                elif matrix_id == 1:
                    old_val = V_arr[row, col]
                else:
                    old_val = W_arr[row, col]
                
                # Try small changes: +/- 0.5, +/- 1
                # Since we want integer/half-integer, we step by 0.5
                delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + delta
                
                # Apply change
                if matrix_id == 0:
                    U_arr[row, col] = new_val
                    U_3d[row // 3, row % 3, col] = new_val
                elif matrix_id == 1:
                    V_arr[row, col] = new_val
                    V_3d[row // 3, row % 3, col] = new_val
                else:
                    W_arr[row, col] = new_val
                    W_3d[row // 3, row % 3, col] = new_val
                
                # Recompute error
                T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U_3d, V_3d, W_3d)
                new_error = np.sum((T - 1.0) ** 2)
                
                if new_error < error:
                    error = new_error
                    if error < best_error:
                        best_error = error
                        best_U = U_arr.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                else:
                    # Revert
                    if matrix_id == 0:
                        U_arr[row, col] = old_val
                        U_3d[row // 3, row % 3, col] = old_val
                    elif matrix_id == 1:
                        V_arr[row, col] = old_val
                        V_3d[row // 3, row % 3, col] = old_val
                    else:
                        W_arr[row, col] = old_val
                        W_3d[row // 3, row % 3, col] = old_val
            
            # If we found a good candidate in this trial, verify it
            if best_error < 1e-9:
                # Re-verify with the best lists
                if verify_333_numpy(best_U, best_V, best_W):
                    return best_U, best_V, best_W, R

        return None

    def schoolbook_333():
        R = 27
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, R

    def schoolbook_331():
        R = 9
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(3)] # m*p = 3*1 = 3
        W = [[0]*R for _ in range(3)] # p*n = 1*3 = 3
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(1):
                    U[i*3+j][col] = 1
                    V[j*1+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, R

    # Try to find better 333 decomposition
    # Time budget for search: 60s.
    # We try rank 23, 24, 25, 26.
    decomp_333 = None
    search_time_left = 60
    
    for R_target in [23, 24, 25, 26]:
        if time.time() + 10 > deadline:
            break
        t_start = time.time()
        decomp_333 = search_333_decomp(R_target, time_limit=15)
        if decomp_333:
            print(f"Found 333 decomp with rank {R_target}")
            break
        # If not found, continue to next rank or time out
    
    if decomp_333 is None:
        decomp_333 = schoolbook_333()
        print("Falling back to schoolbook 333 (rank 27)")
        
    U1, V1, W1, R1 = decomp_333
    
    # Get 3x3x1 decomp
    U2, V2, W2, R2 = schoolbook_331()
    
    # Construct 3x3x7
    # Part 1: 3x3x3 (cols 0-2 of B)
    # Part 2: 3x3x3 (cols 3-5 of B)
    # Part 3: 3x3x1 (col 6 of B)
    
    Total_R = R1 + R1 + R2
    
    U_final = [[0]*Total_R for _ in range(9)]
    V_final = [[0]*Total_R for _ in range(21)] # 3*7
    W_final = [[0]*Total_R for _ in range(21)] # 7*3
    
    # Fill Part 1 (k=0,1,2)
    for r in range(R1):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][r] = U1[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_final[j*7+k][r] = V1[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_final[k*3+i][r] = W1[k*3+i][r]
                
    # Fill Part 2 (k=3,4,5)
    offset1 = R1
    for r in range(R1):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][offset1 + r] = U1[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_final[j*7 + (k+3)][offset1 + r] = V1[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_final[(k+3)*3 + i][offset1 + r] = W1[k*3+i][r]
                
    # Fill Part 3 (k=6)
    offset2 = R1 + R1
    for r in range(R2):
        for i in range(3):
            for j in range(3):
                U_final[i*3+j][offset2 + r] = U2[i*3+j][r]
        for j in range(3):
            for k in range(1):
                V_final[j*7 + 6][offset2 + r] = V2[j*1+k][r]
        for k in range(1):
            for i in range(3):
                W_final[6*3 + i][offset2 + r] = W2[k*3+i][r]
                
    return {"U": U_final, "V": V_final, "W": W_final}
