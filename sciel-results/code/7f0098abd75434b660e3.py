import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Rank-12 decomposition:
    # For each (i, k) pair where i is row of A (and row of C), k is col of B (and col of C):
    # We need sum_j A[i,j] * B[j,k] to appear in C[i,k]
    # Rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If we set for each (i,k): r = i*p + k
    #   U[i*m+j, r] = 1 for all j
    #   V[j*p+k, r] = 1 for all j  
    #   W[k*n+i, r] = 1
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_j (1*1*1) = m = 4
    # That's not right. We need exactly 1 for the (i,j,k) entry.
    # 
    # Actually the tensor T[i,j,k] = 1 for all i,j,k.
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Let me reconsider. The standard matrix multiplication tensor has:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entries are T[i,j,k] = 1 for all valid i,j,k.
    #
    # A rank-1 term contributes U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to T[i,j,k]
    #
    # For the trivial decomposition: rank = n*m*p, with one term per (i,j,k)
    # U[i*m+j, i*m*p + j*p + k] = 1, V[j*p+k, ...] = 1, W[k*n+i, ...] = 1
    #
    # The current seed has rank 44. Let me try to find a better one.
    #
    # Key insight: For fixed i and k, we need sum_j A[i,j]B[j,k] in C[i,k].
    # The tensor is T[i,j,k] = 1. We need sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    #
    # If we fix i,k and let r index over j, we could have:
    # For each (i,k), use m terms (one per j):
    #   U[i*m+j, r_{i,k,j}] = 1, V[j*p+k, r_{i,k,j}] = 1, W[k*n+i, r_{i,k,j}] = 1
    # This gives rank = n*m*p = 48. That's the trivial one.
    #
    # To reduce rank, we need to share terms across different (i,k) pairs.
    #
    # Let me try the approach from the seed: block decomposition.
    # Split into 2x2 blocks of columns/rows.
    
    # Use the known rank-44 solution from the seed, then try to improve
    
    rank_init = 44
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)
    
    # Local 3x2x2 decomposition (Rank 11)
    U_loc = np.zeros((6, 11), dtype=np.float64)
    V_loc = np.zeros((4, 11), dtype=np.float64)
    W_loc = np.zeros((6, 11), dtype=np.float64)
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # 0: A(:, 0:1) * B(0:1, 0:1) -> C(:, 0:1)
    col_offset = 0
    for r in range(3):
        for c in range(2):
            U[(r) * m + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(r) * p + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 1: A(:, 2:3) * B(2:3, 0:1) -> C(:, 0:1)
    col_offset = 11
    for r in range(3):
        for c in range(2):
            U[(r) * m + 2 + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(2 + r) * p + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 2: A(:, 0:1) * B(0:1, 2:3) -> C(:, 2:3)
    col_offset = 22
    for r in range(3):
        for c in range(2):
            U[(r) * m + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(r) * p + 2 + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(2 + c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]

    # 3: A(:, 2:3) * B(2:3, 2:3) -> C(:, 2:3)
    col_offset = 33
    for r in range(3):
        for c in range(2):
            U[(r) * m + 2 + c, col_offset:col_offset+11] += U_loc[r * 2 + c, :]
    for r in range(2):
        for c in range(2):
            V[(2 + r) * p + 2 + c, col_offset:col_offset+11] += V_loc[r * 2 + c, :]
    for r in range(3):
        for c in range(2):
            W[(2 + c) * n + r, col_offset:col_offset+11] += W_loc[r * 2 + c, :]
    
    def check_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False, s
        return True, 0.0
    
    ok, _ = check_decomp(U, V, W, n, m, p)
    if not ok:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Try greedy rank reduction
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank_init
    
    while time.time() < deadline - 5:
        success = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
            mask = [c for c in range(current_rank) if c != col_to_remove]
            test_U = current_U[:, mask].copy()
            test_V = current_V[:, mask].copy()
            test_W = current_W[:, mask].copy()
            
            ok_test, _ = check_decomp(test_U, test_V, test_W, n, m, p)
            if ok_test:
                current_U, current_V, current_W = test_U, test_V, test_W
                current_rank = current_rank - 1
                success = True
                if current_rank < best_rank:
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    best_rank = current_rank
                break
        
        if not success:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
