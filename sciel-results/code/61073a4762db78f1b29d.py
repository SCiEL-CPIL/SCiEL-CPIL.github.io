import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank-56 Strassen decomposition for 2x4 * 4x8
    # This uses 8 blocks of 2x2 * 2x2 Strassen (rank 7 each)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard 2x2 Strassen decomposition
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
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val
    
    # 8 blocks: split 4x8 into 4 blocks of 2x8 (each split into 2 blocks of 2x2)
    # Actually: 2x4 * 4x8 = sum over j_blocks of 2x2 * 2x8
    # Split the 4 columns of A into 2 blocks of 2, and 8 columns of B into 4 blocks of 2
    # So 2 * 4 = 8 blocks of 2x2 * 2x2
    
    # Block (i_off, j_off, k_off) where i_off in {0}, j_off in {0,1}, k_off in {0,1,2,3}
    for j_off in range(2):  # 2 blocks of width 2 in the middle dimension
        for k_off in range(4):  # 4 blocks of width 2 in the output dimension
            base_rank = (j_off * 4 + k_off) * 7
            add_block(base_rank, 0, j_off * 2, k_off * 2)
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        # Use vectorized numpy operations for speed
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape to (n, m, R) etc?
        # U is indexed by (i, j). V by (j, k). W by (k, i).
        
        # Compute for all i, j, k
        # This is O(n*m*p*R)
        # n=2, m=4, p=8, R=56. 2*4*8*56 = 3584 ops. Very fast.
        
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :rank_val]
                for k in range(p):
                    v_row = V_arr[j*p+k, :rank_val]
                    w_row = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_row, v_row * w_row)
                    
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=1e-10)
    
    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np, rank):
        # Fallback: return the original Strassen decomposition
        return {"U": U, "V": V, "W": W}
    
    # Now attempt rank reduction by trying to zero out columns
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    # Repeatedly try to zero out a column
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
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
    
    # Convert back to lists
    if best_rank < rank:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        U_final = [row[:best_rank] for row in U_final]
        V_final = [row[:best_rank] for row in V_final]
        W_final = [row[:best_rank] for row in W_final]
        
        for mat in [U_final, V_final, W_final]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-10:
                        mat[i][j] = rounded
        
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
