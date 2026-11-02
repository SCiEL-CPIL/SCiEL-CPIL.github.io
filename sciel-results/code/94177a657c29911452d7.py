import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 115

    def build_schoolbook(n, m, p):
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    col = i * m * p + j * p + k
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
        return U, V, W

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        """Verify that U,V,W exactly decompose the multiplication tensor."""
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
            return False
        if U_mat.shape[1] != r or V_mat.shape[1] != r or W_mat.shape[1] != r:
            return False
        
        # Check required entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val - 1) > tol:
                        return False
        
        # Check all other entries are 0
        # For efficiency, we can sample or check all. For small sizes, check all.
        # Total checks: n*m * m*p * p*n = n^2 m^2 p^2. For 3,3,7: 9*21*21 = 3969. Very fast.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if i == ii and j == j2 and k == kk:
                        continue
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val) > tol:
                        return False
        return True

    # Build schoolbook for the full 3x3x7 tensor
    U, V, W = build_schoolbook(n, m, p)
    
    # Verify schoolbook
    if not verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook

    # Strategy: Exploit block-linearity.
    # The tensor is T_{i,j,k} for i in 0..2, j in 0..2, k in 0..6.
    # We can view this as a direct sum of two tensors:
    # T1: k in 0..2 (3x3x3 part)
    # T2: k in 3..6 (3x3x4 part)
    # However, the indices a,b,c couple i,j,k.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    #
    # If we split k into k1 (0..2) and k2 (3..6), let k2 = 3 + l where l in 0..3.
    # The tensor is not a simple direct sum in the U,V,W matrix space because
    # the column index in U,V,W depends on the specific (i,j,k) triple.
    # But we can try to optimize the 3x3x4 part (k=3..6) and keep the 3x3x3 part (k=0..2) fixed?
    # No, because the rank is shared.
    #
    # Better approach:
    # 1. Find a low-rank decomposition for 3x3x3 (Strassen rank 23).
    # 2. Find a low-rank decomposition for 3x3x4.
    # 3. Combine them.
    #
    # Let's first try to improve the 3x3x3 part using known Strassen decomposition.
    # Strassen's 7x7 matrix multiplication algorithm gives rank 23 for 3x3x3?
    # Actually, Strassen's algorithm is for 2x2x2 (rank 7).
    # For 3x3x3, the rank is known to be at most 23 (Strassen 1969, later improved?).
    # Actually, the rank of 3x3x3 is 23.
    
    # Let's hardcode a known 23-rank decomposition for 3x3x3.
    # Source: https://www.researchgate.net/publication/220573859_The_rank_of_the_3x3x3_tensor
    # Or we can search for it.
    # Since we have time, let's try to find a 23-rank decomposition for 3x3x3 via search?
    # That might be too hard.
    #
    # Alternative: Use the "border rank" or other known results?
    # No, we need exact rank.
    #
    # Let's try a randomized search for the 3x3x4 part.
    # The schoolbook rank for 3x3x4 is 36.
    # Can we do better?
    #
    # Let's try to optimize the whole 3x3x7 tensor using a more aggressive random search.
    # We will use the "peeling" method but with a better initial guess or perturbation.
    #
    # Actually, let's try to find a decomposition for the 3x3x4 part specifically.
    # If we can find a rank-r decomposition for 3x3x4, and we have a rank-23 decomposition for 3x3x3,
    # the total rank is 23 + r.
    # But wait, the direct sum of two tensor decompositions is a valid decomposition of the direct sum of the tensors.
    # The 3x3x7 tensor is the direct sum of the 3x3x3 tensor (k=0,1,2) and the 3x3x4 tensor (k=3,4,5,6).
    # Yes!
    #
    # So the problem decomposes into:
    # 1. Find min rank of 3x3x3 multiplication tensor.
    # 2. Find min rank of 3x3x4 multiplication tensor.
    # 3. Sum the ranks.
    #
    # Known rank of 3x3x3 is 23.
    # Let's hardcode a 23-rank decomposition for 3x3x3.
    # I will generate one using a known construction or search.
    #
    # For now, let's assume we can find a 23-rank decomposition for 3x3x3.
    # And let's try to find a rank < 36 decomposition for 3x3x4.
    #
    # Let's write a function to find a low-rank decomposition for a given (n,m,p) using random search.
    # We will use this for 3x3x4.

    def find_decomposition(n, m, p, target_rank, max_time=10):
        """
        Try to find a rank-target_rank decomposition for n x m * p multiplication.
        Returns (U, V, W, rank) or None.
        """
        start = time.time()
        np.random.seed(int(start * 1000) % 2**31)
        
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        max_stuck = 100 # Restart if we fail 100 random removal attempts in a row
        
        while time.time() - start < max_time:
            # Restart with schoolbook
            U_cur, V_cur, W_cur = build_schoolbook(n, m, p)
            cur_rank = n * m * p
            
            stuck_counter = 0
            
            while cur_rank > target_rank and time.time() - start < max_time:
                if stuck_counter > max_stuck:
                    break # Restart outer loop
                
                # Pick a random column
                col_to_try = np.random.randint(0, cur_rank)
                
                # Create mask
                mask = np.ones(cur_rank, dtype=bool)
                mask[col_to_try] = False
                
                U_cand = U_cur[:, mask]
                V_cand = V_cur[:, mask]
                W_cand = W_cur[:, mask]
                
                if verify(U_cand, V_cand, W_cand, n, m, p):
                    U_cur = U_cand
                    V_cur = V_cand
                    W_cur = W_cand
                    cur_rank -= 1
                    stuck_counter = 0
                    # Update best if this is better
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                else:
                    stuck_counter += 1
                    
            # If we reached target_rank, we can break early?
            # The outer loop will check time and break if not improved or if time is up.
            # If cur_rank == target_rank, we found a solution.
            # We can break the outer loop if we hit the target.
            if cur_rank <= target_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None

    # Let's try to find a decomposition for 3x3x4
    # We will allocate 80 seconds for this search.
    t_start = time.time()
    result_334 = find_decomposition(3, 3, 4, 30, max_time=80)
    t_end = time.time()
    
    if result_334:
        U4, V4, W4, rank_334 = result_334
        print(f"Found 3x3x4 decomposition with rank {rank_334}")
    else:
        # Fall back to schoolbook
        U4, V4, W4 = build_schoolbook(3, 3, 4)
        rank_334 = 36
        print("Failed to find better 3x3x4 decomposition, using schoolbook")

    # Now we need a 3x3x3 decomposition.
    # We know rank 23 is possible.
    # Let's try to find it or use a known one.
    # If we can't find it quickly, use schoolbook (rank 27).
    
    # Let's try to find a 3x3x3 decomposition with rank 23.
    # Allocate 30 seconds.
    t_start = time.time()
    result_333 = find_decomposition(3, 3, 3, 23, max_time=30)
    t_end = time.time()
    
    if result_333:
        U3, V3, W3, rank_333 = result_333
        print(f"Found 3x3x3 decomposition with rank {rank_333}")
    else:
        # Fall back to schoolbook
        U3, V3, W3 = build_schoolbook(3, 3, 3)
        rank_333 = 27
        print("Failed to find better 3x3x3 decomposition, using schoolbook")

    # Combine the decompositions.
    # The 3x3x7 tensor is the direct sum of 3x3x3 (k=0,1,2) and 3x3x4 (k=3,4,5,6).
    # We need to construct the full U, V, W matrices.
    
    total_rank = rank_333 + rank_334
    
    U_full = np.zeros((3 * 3, total_rank), dtype=float)
    V_full = np.zeros((3 * 7, total_rank), dtype=float)
    W_full = np.zeros((7 * 3, total_rank), dtype=float)
    
    # Copy 3x3x3 part (k=0,1,2)
    # U3 is 9x23 (or 27).
    # V3 is 3x3 = 9x23.
    # W3 is 3x3 = 9x23.
    
    U_full[:, :rank_333] = U3
    # V3 corresponds to j in 0..2, k in 0..2.
    # In V_full, the rows are indexed by j*p + k.
    # For the 3x3x3 part, k is 0,1,2.
    # So rows 0..8 of V_full (j=0, k=0..2; j=1, k=0..2; j=2, k=0..2) correspond to V3.
    V_full[0:9, :rank_333] = V3
    # W3 corresponds to k in 0..2, i in 0..2.
    # In W_full, the rows are indexed by k*n + i.
    # For k=0,1,2, rows are 0..8.
    W_full[0:9, :rank_333] = W3
    
    # Copy 3x3x4 part (k=3,4,5,6)
    # U4 is 9x36 (or less).
    # U_full[:, rank_333:] = U4
    
    U_full[:, rank_333:] = U4
    
    # V4 corresponds to j in 0..2, k in 0..3 (local index).
    # In the full tensor, k is 3,4,5,6.
    # So rows in V_full are j*7 + k.
    # j=0: k=3,4,5,6 -> rows 3,4,5,6
    # j=1: k=3,4,5,6 -> rows 10,11,12,13
    # j=2: k=3,4,5,6 -> rows 17,18,19,20
    
    # V4 is 12x36 (3*4 rows).
    # We need to map V4 rows to V_full rows.
    for j in range(3):
        for l in range(4):
            k_local = l
            k_full = 3 + l
            row_v4 = j * 4 + l
            row_v_full = j * 7 + k_full
            V_full[row_v_full, rank_333:] = V4[row_v4, :]
            
    # W4 corresponds to k_local in 0..3, i in 0..2.
    # In the full tensor, k is 3,4,5,6.
    # So rows in W_full are k*n + i.
    # k=3: i=0,1,2 -> rows 9,10,11
    # k=4: i=0,1,2 -> rows 12,13,14
    # k=5: i=0,1,2 -> rows 15,16,17
    # k=6: i=0,1,2 -> rows 18,19,20
    
    # W4 is 12x36 (4*3 rows).
    for l in range(4):
        k_local = l
        k_full = 3 + l
        for i in range(3):
            row_w4 = l * 3 + i
            row_w_full = k_full * 3 + i
            W_full[row_w_full, rank_333:] = W4[row_w4, :]

    # Verify the full decomposition
    if verify(U_full, V_full, W_full, 3, 3, 7):
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = total_rank
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank_schoolbook

    # Convert to half-integers if possible
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-9):
            return (rounded / 2).tolist()
        return mat.tolist()

    return {"U": to_half_integer(best_U), "V": to_half_integer(best_V), "W": to_half_integer(best_W)}
