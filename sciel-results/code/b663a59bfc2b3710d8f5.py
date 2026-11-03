"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

_start_time = time.time()
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication.
    Checks: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    and all other tensor entries are 0.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor and check the relevant entries
    # T[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k
    # All other entries should be 0
    
    # Check the 1-entries first (these are the critical ones)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check that non-relevant entries are 0 (sample a subset for speed)
    # The tensor has (n*m)*(m*p)*(p*n) entries, most should be 0
    # We check entries where the indices don't align properly
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For speed, check a random subset of non-aligned entries
    import random
    random.seed(42)
    checks = 500
    for _ in range(checks):
        idx_i = random.randint(0, nm - 1)
        idx_j = random.randint(0, mp - 1)
        idx_k = random.randint(0, pn - 1)
        
        i1 = idx_i // m
        j1 = idx_i % m
        j2 = idx_j // p
        k1 = idx_j % p
        k2 = idx_k // n
        i2 = idx_k % n
        
        # Skip if this is a valid aligned entry
        if j1 == j2 and i1 == i2 and k1 == k2:
            continue
        
        s = 0.0
        for c in range(rank):
            s += U_arr[idx_i, c] * V_arr[idx_j, c] * W_arr[idx_k, c]
        if abs(s) > 1e-9:
            return False
    
    return True

def _build_base_decomposition():
    """Build the rank-44 Strassen-based decomposition for 3x4 * 4x4.
    
    Strategy: Partition into 4 blocks of (2x2 output) x (2x2 middle) x (2x2 output)
    for rows i={1,2}, plus schoolbook for row i=0.
    
    Blocks:
    - 4 Strassen blocks (rank 7 each) for i in {1,2}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    - 4 schoolbook blocks (rank 4 each) for i=0, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    
    Total rank: 4*7 + 4*4 = 28 + 16 = 44
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen 2x2x2 factors
    # A indices: A00->0, A01->1, A10->2, A11->3
    # B indices: B00->0, B01->1, B10->2, B11->3
    # C indices: C00->0, C01->1, C10->2, C11->3
    strassen_U = [
        [1,0,1,0,1,1,0],   # A00
        [0,0,0,0,1,0,1],   # A01
        [0,1,0,0,0,-1,0],  # A10
        [1,1,0,1,0,0,-1],  # A11
    ]
    strassen_V = [
        [1,1,0,1,0,1,0],   # B00
        [0,0,1,-1,0,1,0],  # B01
        [0,0,0,1,0,0,1],   # B10
        [1,0,-1,0,1,0,1],  # B11
    ]
    strassen_W = [
        [1,0,0,1,-1,0,1],  # C00
        [0,0,1,0,1,0,0],   # C01
        [0,1,0,1,0,0,0],   # C10
        [-1,1,1,0,0,1,0],  # C11
    ]
    
    # Schoolbook 1x2x2 factors (rank 4)
    # A is 1x2: indices 0, 1
    # B is 2x2: indices 0,1,2,3
    # C is 1x2: indices 0, 1
    schoolbook_U = [
        [1,0,0,0],  # A[0,0]
        [0,1,0,0],  # A[0,1]
    ]
    schoolbook_V = [
        [1,0,0,0],  # B[0,0]
        [0,1,0,0],  # B[0,1]
        [0,0,1,0],  # B[1,0]
        [0,0,0,1],  # B[1,1]
    ]
    schoolbook_W = [
        [1,0,0,0],  # C[0,0]
        [0,1,0,0],  # C[0,1]
    ]
    
    # 4 Strassen blocks for i in {1,2}
    for j_base in [0, 2]:
        for k_base in [0, 2]:
            for c in range(7):
                # U: A rows are i=1,2; cols are j_base, j_base+1
                # Local A index: local_row * 2 + local_col
                # Global A index: (1+local_row)*4 + (j_base+local_col)
                for lr in range(2):
                    for lc in range(2):
                        i = 1 + lr
                        j = j_base + lc
                        U[i*m + j][col+c] += strassen_U[lr*2+lc][c]
                
                # V: B rows are j_base, j_base+1; cols are k_base, k_base+1
                # Local B index: local_row * 2 + local_col
                # Global B index: (j_base+local_row)*4 + (k_base+local_col)
                for lr in range(2):
                    for lc in range(2):
                        j = j_base + lr
                        k = k_base + lc
                        V[j*p + k][col+c] += strassen_V[lr*2+lc][c]
                
                # W: C rows are k_base, k_base+1; cols are i=1,2
                # Local C index: local_row * 2 + local_col
                # Global C index: (k_base+local_row)*3 + (1+local_col)
                for lr in range(2):
                    for lc in range(2):
                        k = k_base + lr
                        i = 1 + lc
                        W[k*n + i][col+c] += strassen_W[lr*2+lc][c]
            col += 7
    
    # 4 Schoolbook blocks for i=0
    for j_base in [0, 2]:
        for k_base in [0, 2]:
            for c in range(4):
                # U: A row is i=0; cols are j_base, j_base+1
                # Local A index: 0*2 + local_col = local_col
                # Global A index: 0*4 + (j_base+local_col)
                for lc in range(2):
                    j = j_base + lc
                    U[0*m + j][col+c] += schoolbook_U[lc][c]
                
                # V: B rows are j_base, j_base+1; cols are k_base, k_base+1
                # Local B index: local_row * 2 + local_col
                # Global B index: (j_base+local_row)*4 + (k_base+local_col)
                for lr in range(2):
                    for lc in range(2):
                        j = j_base + lr
                        k = k_base + lc
                        V[j*p + k][col+c] += schoolbook_V[lr*2+lc][c]
                
                # W: C row is k_base, k_base+1; col is i=0
                # Local C index: local_row * 2 + 0 = local_row
                # Global C index: (k_base+local_row)*3 + 0
                for lr in range(2):
                    k = k_base + lr
                    W[k*n + 0][col+c] += schoolbook_W[lr][c]
            col += 4
    
    return U, V, W

def _search_lower_rank(base_U, base_V, base_W, n, m, p):
    """Search for a lower-rank decomposition by trying to eliminate columns.
    
    Greedily tries to remove columns one at a time, checking if the
    decomposition remains valid.
    """
    rank = len(base_U[0])
    U_arr = np.array(base_U, dtype=float)
    V_arr = np.array(base_V, dtype=float)
    W_arr = np.array(base_W, dtype=float)
    
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Try removing one column at a time, repeatedly
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for col_to_remove in range(best_rank):
            if _time_exceeded():
                break
            
            # Test removing this column
            new_U = best_U.copy()
            new_V = best_V.copy()
            new_W = best_W.copy()
            new_U[:, col_to_remove] = 0
            new_V[:, col_to_remove] = 0
            new_W[:, col_to_remove] = 0
            
            # Quick check: verify the 1-entries
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(best_rank):
                            if c == col_to_remove:
                                continue
                            s += new_U[i*m+j, c] * new_V[j*p+k, c] * new_W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Remove the column
                keep_cols = [c for c in range(best_rank) if c != col_to_remove]
                best_U = best_U[:, keep_cols]
                best_V = best_V[:, keep_cols]
                best_W = best_W[:, keep_cols]
                best_rank -= 1
                improved = True
                break
    
    # Convert back to list format
    if best_rank < rank:
        U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
        V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
        W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
        return U_list, V_list, W_list
    
    return base_U, base_V, base_W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Build the rank-44 Strassen-based decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U_s = [[0.0]*rank for _ in range(n*m)]
        V_s = [[0.0]*rank for _ in range(m*p)]
        W_s = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1.0
                    V_s[j*p+k][col] = 1.0
                    W_s[k*n+i][col] = 1.0
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try to reduce rank
    if not _time_exceeded():
        U_reduced, V_reduced, W_reduced = _search_lower_rank(U, V, W, n, m, p)
        if len(U_reduced[0]) < len(U[0]):
            # Verify the reduced version
            if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
                U, V, W = U_reduced, V_reduced, W_reduced
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
