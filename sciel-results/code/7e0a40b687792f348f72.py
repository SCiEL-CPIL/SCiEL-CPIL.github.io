import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2 patterns
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in sU[r]:
                li, lj = divmod(idx, 2)
                gi = li + i_off
                gj = lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in sV[r]:
                lj, lk = divmod(idx, 2)
                gj = lj + j_off
                gk = lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in sW[r]:
                lk, li = divmod(idx, 2)
                gk = lk + k_off
                gi = li + i_off
                W[gk * 2 + gi][cr] += val
    
    # Sub-problem 1: A_L (i:0-1, j:0-1) * B_T (j:0-1, k:0-7)
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    
    # Sub-problem 2: A_R (i:0-1, j:2-3) * B_B (j:2-3, k:0-7)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to reduce rank by checking for linear dependencies
    keep = list(range(rank))
    reduced = False
    
    # Greedy removal: try to remove each column if possible
    for _ in range(20):  # Limit iterations to stay within time
        if time.time() > deadline - 5:
            break
        
        removed_any = False
        for pos in range(len(keep)):
            if time.time() > deadline - 5:
                break
            
            r = keep[pos]
            others = [c for c in keep if c != r]
            if not others:
                break
            
            # Check if column r is in the span of others for all three matrices
            # with the SAME coefficients
            
            # Solve for coefficients in U
            U_others = U_np[:, others]
            U_r = U_np[:, r]
            
            # Check if U_r is in span of U_others
            rank_others = np.linalg.matrix_rank(U_others)
            rank_with_r = np.linalg.matrix_rank(np.column_stack([U_others, U_r]))
            
            if rank_with_r > rank_others:
                continue  # U_r not in span
            
            # Solve for coefficients
            try:
                # Use least squares to find coefficients
                coeffs_U, _, _, _ = np.linalg.lstsq(U_others, U_r, rcond=None)
            except:
                continue
            
            # Verify the same coefficients work for V and W
            V_others = V_np[:, others]
            V_r = V_np[:, r]
            V_pred = V_others @ coeffs_U
            
            W_others = W_np[:, others]
            W_r = W_np[:, r]
            W_pred = W_others @ coeffs_U
            
            # Check if predictions match (within tolerance)
            tol = 1e-10
            if np.allclose(V_pred, V_r, atol=tol) and np.allclose(W_pred, W_r, atol=tol):
                # Remove column r
                keep.pop(pos)
                removed_any = True
                reduced = True
                break
        
        if not removed_any:
            break
    
    # Build final result with reduced rank
    final_rank = len(keep)
    U_final = [[0] * final_rank for _ in range(n * m)]
    V_final = [[0] * final_rank for _ in range(m * p)]
    W_final = [[0] * final_rank for _ in range(p * n)]
    
    for new_idx, old_idx in enumerate(keep):
        for i in range(n * m):
            U_final[i][new_idx] = U[i][old_idx]
        for j in range(m * p):
            V_final[j][new_idx] = V[j][old_idx]
        for k in range(p * n):
            W_final[k][new_idx] = W[k][old_idx]
    
    return {"U": U_final, "V": V_final, "W": W_final}
