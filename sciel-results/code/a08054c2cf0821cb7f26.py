"""Matrix multiplication 2x4x8 - rank minimization via block decomposition + greedy reduction."""
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    n, m, p = 2, 4, 8
    deadline = _START_TIME + 110  # safety margin

    # ============================================================
    # Build the rank-56 Strassen decomposition
    # ============================================================
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # For a 2x2 block: C = A*B where A,B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients for each Strassen term (row, col in 2x2 block)
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},              # M3
        {(1,1): 1},              # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1},  # M7
    ]
    
    # V coefficients for each Strassen term (row, col in 2x2 block)
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},              # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},              # M5
        {(0,0): 1, (0,1): 1},   # M6
        {(1,0): 1, (1,1): 1},   # M7
    ]
    
    # W coefficients for each Strassen term (row=k, col=i in 2x2 block)
    # C00 = M1 - M4 + M5 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # M1: C00+=1, C11+=1
        {(0,1): 1, (1,1): -1},  # M2: C01+=1, C11-=-1 -> C11+=1? Wait.
        # Let me re-derive:
        # C00 = M1 - M4 + M5 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # W[k*n+i] where k is row of C, i is col of C
        # So W[0*2+0] = C00 coeff, W[0*2+1] = C01 coeff, etc.
    ]
    
    # Redo W properly:
    # For term r, W[(k)*n+i] = coefficient of M_r in C[k,i]
    # C[0,0] = M1 - M4 + M5 - M6 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # M1: C00+=1, C11+=1
        {(1,0): 1, (1,1): -1},  # M2: C10+=1, C11+=-1
        {(0,1): 1, (1,1): 1},   # M3: C01+=1, C11+=1
        {(0,0): -1, (1,0): 1},  # M4: C00+=-1, C10+=1
        {(0,0): 1, (0,1): 1},   # M5: C00+=1, C01+=1
        {(0,0): -1},            # M6: C00+=-1
        {(0,0): 1, (0,1): 1},   # M7: C00+=1, C01+=1
    ]
    
    # Wait, let me verify M7: C00 += M7, C01 += M7? 
    # C00 = M1 - M4 + M5 - M6 + M7 -> yes M7 contributes +1 to C00
    # C01 = M3 + M5 -> M7 does NOT contribute to C01
    # C11 = M1 - M2 + M3 + M7 -> M7 contributes +1 to C11
    
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # M1: C00+=1, C11+=1
        {(1,0): 1, (1,1): -1},  # M2: C10+=1, C11+=-1
        {(0,1): 1, (1,1): 1},   # M3: C01+=1, C11+=1
        {(0,0): -1, (1,0): 1},  # M4: C00+=-1, C10+=1
        {(0,0): 1, (0,1): 1},   # M5: C00+=1, C01+=1
        {(0,0): -1},            # M6: C00+=-1
        {(0,0): 1, (1,1): 1},   # M7: C00+=1, C11+=1
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj, cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk, cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi, cr] += val
    
    # 8 blocks: j0 in {0,2}, k0 in {0,2,4,6}
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # ============================================================
    # Verification
    # ============================================================
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
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=1e-10)
    
    # Verify initial decomposition
    if not is_valid(U, V, W, rank):
        # Fallback to naive rank-64
        U_naive = np.zeros((n*m, n*m*p), dtype=float)
        V_naive = np.zeros((m*p, n*m*p), dtype=float)
        W_naive = np.zeros((p*n, n*m*p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, idx] = 1.0
                    V_naive[j*p+k, idx] = 1.0
                    W_naive[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # ============================================================
    # Greedy rank reduction: try to zero out columns
    # ============================================================
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Try zeroing out column r
            test_U = current_U.copy()
            test_V = current_V.copy()
            test_W = current_W.copy()
            test_U[:, r] = 0
            test_V[:, r] = 0
            test_W[:, r] = 0
            
            if is_valid(test_U, test_V, test_W, current_rank):
                mask = [x for x in range(current_rank) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
        
        if time.time() > deadline - 5:
            break
    
    # ============================================================
    # Try to reduce rank by finding linear dependencies
    # ============================================================
    if _time_left() > 20:
        # For each pair of columns, check if one is a linear combination of others
        # This is expensive, so we do it selectively
        current_rank = best_rank
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        
        # Try removing each column and checking if the tensor can be reconstructed
        # by scaling the remaining columns
        for r_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Try to see if column r_remove can be expressed as a combination of others
            # This is a hard problem, so we try a simple heuristic:
            # Check if zeroing it out and adjusting others works
            
            # Alternative: try random linear combinations
            import random
            random.seed(42 + r_remove)
            
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                test_U = current_U.copy()
                test_V = current_V.copy()
                test_W = current_W.copy()
                
                # Try to merge column r_remove into another column
                r_target = random.randint(0, current_rank - 1)
                if r_target == r_remove:
                    r_target = (r_target + 1) % current_rank
                
                # Try to add a multiple of column r_remove to r_target
                for alpha in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25]:
                    test_U[:, r_target] += alpha * test_U[:, r_remove]
                    test_V[:, r_target] += alpha * test_V[:, r_remove]
                    test_W[:, r_target] += alpha * test_W[:, r_remove]
                    
                    # Zero out r_remove
                    test_U[:, r_remove] = 0
                    test_V[:, r_remove] = 0
                    test_W[:, r_remove] = 0
                    
                    if is_valid(test_U, test_V, test_W, current_rank):
                        # Remove column r_remove
                        mask = [x for x in range(current_rank) if x != r_remove]
                        test_U = test_U[:, mask]
                        test_V = test_V[:, mask]
                        test_W = test_W[:, mask]
                        
                        if current_rank - 1 < best_rank:
                            best_rank = current_rank - 1
                            best_U = test_U.copy()
                            best_V = test_V.copy()
                            best_W = test_W.copy()
                            current_rank -= 1
                            current_U = test_U.copy()
                            current_V = test_V.copy()
                            current_W = test_W.copy()
                        break
    
    # ============================================================
    # Convert to lists and return
    # ============================================================
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
