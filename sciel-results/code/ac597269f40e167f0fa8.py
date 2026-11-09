<<<<<<< SEARCH
    # Verify correctness with multiple deterministic tests
    try:
        # Test 1: Fixed matrix
        A_test1 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        B_test1 = [[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]]
        
        # Test 2: Identity and Zero
        A_test2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        B_test2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        
        # Test 3: Random-ish fixed
        A_test3 = [[1,1,1,1],[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        B_test3 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        
        tests = [(A_test1, B_test1), (A_test2, B_test2), (A_test3, B_test3)]
        
        for A_t, B_t in tests:
            # Compute A*B using schoolbook
            AB = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        AB[i][j] += A_t[i][k] * B_t[k][j]
            
            # Compute using decomposition
            AB_decomp = [[0]*4 for _ in range(4)]
            for r in range(49):
                u_val = 0
                for i in range(4):
                    for j in range(4):
                        u_val += U[i*4+j][r] * A_t[i][j]
                v_val = 0
                for j in range(4):
                    for k in range(4):
                        v_val += V[j*4+k][r] * B_t[j][k]
                scalar = u_val * v_val
                for i in range(4):
                    for k in range(4):
                        AB_decomp[i][k] += W[k*4+i][r] * scalar
            
            # Check if they match
            for i in range(4):
                for j in range(4):
                    if AB[i][j] != AB_decomp[i][j]:
                        # Fallback to schoolbook rank 64
                        rank_fb = 64
                        U_fb = [[0]*rank_fb for _ in range(16)]
                        V_fb = [[0]*rank_fb for _ in range(16)]
                        W_fb = [[0]*rank_fb for _ in range(16)]
                        col_fb = 0
                        for i2 in range(4):
                            for j2 in range(4):
                                for k2 in range(4):
                                    U_fb[i2*4+j2][col_fb] = 1
                                    V_fb[j2*4+k2][col_fb] = 1
                                    W_fb[k2*4+i2][col_fb] = 1
                                    col_fb += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    # Verify correctness with multiple deterministic tests
    try:
        # Test 1: Fixed matrix
        A_test1 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        B_test1 = [[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]]
        
        # Test 2: Identity and Zero
        A_test2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        B_test2 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        
        # Test 3: Random-ish fixed
        A_test3 = [[1,1,1,1],[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        B_test3 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        
        tests = [(A_test1, B_test1), (A_test2, B_test2), (A_test3, B_test3)]
        
        for A_t, B_t in tests:
            # Compute A*B using schoolbook
            AB = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        AB[i][j] += A_t[i][k] * B_t[k][j]
            
            # Compute using decomposition
            AB_decomp = [[0]*4 for _ in range(4)]
            for r in range(49):
                u_val = 0
                for i in range(4):
                    for j in range(4):
                        u_val += U[i*4+j][r] * A_t[i][j]
                v_val = 0
                for j in range(4):
                    for k in range(4):
                        v_val += V[j*4+k][r] * B_t[j][k]
                scalar = u_val * v_val
                for i in range(4):
                    for k in range(4):
                        AB_decomp[i][k] += W[k*4+i][r] * scalar
            
            # Check if they match
            for i in range(4):
                for j in range(4):
                    if AB[i][j] != AB_decomp[i][j]:
                        # Fallback to schoolbook rank 64
                        rank_fb = 64
                        U_fb = [[0]*rank_fb for _ in range(16)]
                        V_fb = [[0]*rank_fb for _ in range(16)]
                        W_fb = [[0]*rank_fb for _ in range(16)]
                        col_fb = 0
                        for i2 in range(4):
                            for j2 in range(4):
                                for k2 in range(4):
                                    U_fb[i2*4+j2][col_fb] = 1
                                    V_fb[j2*4+k2][col_fb] = 1
                                    W_fb[k2*4+i2][col_fb] = 1
                                    col_fb += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    # Attempt greedy rank reduction: check if any term is redundant
    # For each term, test if removing it still gives correct results
    # Use multiple random-ish test cases for robustness
    if time.time() < deadline - 20:
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Generate additional test cases
        test_cases = []
        # Use structured pseudo-random matrices
        for seed_idx in range(5):
            A_t = [[(i*4+j+seed_idx*3+1) % 7 - 3 for j in range(4)] for i in range(4)]
            B_t = [[(i*4+j+seed_idx*5+2) % 7 - 3 for j in range(4)] for i in range(4)]
            test_cases.append((A_t, B_t))
        
        # For each candidate term to remove, verify correctness
        # Try removing each term and check against test cases
        best_rank = 49
        best_indices = list(range(49))
        
        # Greedy: try removing one term at a time
        # For efficiency, only try a subset of terms
        terms_to_try = list(range(49))
        
        for term_idx in terms_to_try:
            if time.time() > deadline - 15:
                break
            
            # Check if removing this term still works
            valid = True
            for A_t, B_t in test_cases:
                # Compute expected result
                A_np = np.array(A_t, dtype=float)
                B_np = np.array(B_t, dtype=float)
                expected = A_np @ B_np
                
                # Compute with this term removed
                result = np.zeros((4, 4))
                for r in range(49):
                    if r == term_idx:
                        continue
                    u_val = sum(U_np[i*4+j, r] * A_t[i][j] for i in range(4) for j in range(4))
                    v_val = sum(V_np[j*4+k, r] * B_t[j][k] for j in range(4) for k in range(4))
                    scalar = u_val * v_val
                    for i in range(4):
                        for k in range(4):
                            result[i, k] += W_np[k*4+i, r] * scalar
                
                if not np.allclose(result, expected, atol=1e-9):
                    valid = False
                    break
            
            if valid:
                # Term is redundant, remove it
                best_indices = [i for i in best_indices if i != term_idx]
                best_rank = len(best_indices)
        
        # If we found a lower rank, construct the reduced decomposition
        if best_rank < 49:
            U_red = [[0]*best_rank for _ in range(16)]
            V_red = [[0]*best_rank for _ in range(16)]
            W_red = [[0]*best_rank for _ in range(16)]
            for new_col, old_col in enumerate(best_indices):
                for row in range(16):
                    U_red[row][new_col] = U[row][old_col]
                    V_red[row][new_col] = V[row][old_col]
                    W_red[row][new_col] = W[row][old_col]
            return {"U": U_red, "V": V_red, "W": W_red}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
