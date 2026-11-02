import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 5, 6
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)×[0,m)×[0,p)
    # We need to decompose T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^(n*m), v_r is in R^(m*p), w_r is in R^(p*n)
    # with the convention W[k*n+i, r]
    
    # Generate candidate rank-1 terms
    # Each term is defined by:
    # - U support: subset of (i,j) pairs with values
    # - V support: subset of (j,k) pairs with values  
    # - W support: subset of (k,i) pairs with values
    # The term contributes to T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # For a term to be "valid" for the multiplication tensor, we need:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all i,j,k
    
    # Generate Strassen-style terms for all 2x2x2 sub-blocks
    candidates = []
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Add all individual schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U = {u_idx(i, j): 1}
                V = {v_idx(j, k): 1}
                W = {w_idx(k, i): 1}
                candidates.append((U, V, W))
    
    # Add Strassen terms for all 2x2x2 blocks
    for j1 in range(m - 1):
        j2 = j1 + 1
        for k1 in range(p - 1):
            k2 = k1 + 1
            # m1 = (a00+a11)(b00+b11)
            candidates.append(
                ({u_idx(0,j1): 1, u_idx(1,j2): 1}, 
                 {v_idx(j1,k1): 1, v_idx(j2,k2): 1}, 
                 {w_idx(k1,0): 1, w_idx(k2,1): 1})
            )
            # m2 = (a10+a11)*b00
            candidates.append(
                ({u_idx(1,j1): 1, u_idx(1,j2): 1}, 
                 {v_idx(j1,k1): 1}, 
                 {w_idx(k1,1): 1, w_idx(k2,1): -1})
            )
            # m3 = a00*(b01-b11)
            candidates.append(
                ({u_idx(0,j1): 1}, 
                 {v_idx(j1,k2): 1, v_idx(j2,k2): -1}, 
                 {w_idx(k2,0): 1, w_idx(k2,1): 1})
            )
            # m4 = a11*(b10-b00)
            candidates.append(
                ({u_idx(1,j2): 1}, 
                 {v_idx(j2,k1): 1, v_idx(j1,k1): -1}, 
                 {w_idx(k1,0): 1, w_idx(k1,1): 1})
            )
            # m5 = (a00+a01)*b11
            candidates.append(
                ({u_idx(0,j1): 1, u_idx(0,j2): 1}, 
                 {v_idx(j2,k2): 1}, 
                 {w_idx(k1,0): -1, w_idx(k2,0): 1})
            )
            # m6 = (a10-a00)*(b00+b01)
            candidates.append(
                ({u_idx(1,j1): 1, u_idx(0,j1): -1}, 
                 {v_idx(j1,k1): 1, v_idx(j1,k2): 1}, 
                 {w_idx(k2,1): 1})
            )
            # m7 = (a01-a11)*(b10+b11)
            candidates.append(
                ({u_idx(0,j2): 1, u_idx(1,j2): -1}, 
                 {v_idx(j2,k1): 1, v_idx(j2,k2): 1}, 
                 {w_idx(k1,0): 1})
            )
    
    # Also add some 2x1x2 and 2x2x1 style terms for flexibility
    # 2x1x2: for fixed j, all i and k
    for j in range(m):
        for k1 in range(p - 1):
            k2 = k1 + 1
            # i=0, k1
            candidates.append(({u_idx(0,j): 1}, {v_idx(j,k1): 1}, {w_idx(k1,0): 1}))
            # i=0, k2
            candidates.append(({u_idx(0,j): 1}, {v_idx(j,k2): 1}, {w_idx(k2,0): 1}))
            # i=1, k1
            candidates.append(({u_idx(1,j): 1}, {v_idx(j,k1): 1}, {w_idx(k1,1): 1}))
            # i=1, k2
            candidates.append(({u_idx(1,j): 1}, {v_idx(j,k2): 1}, {w_idx(k2,1): 1}))
    
    # For each candidate, compute which tensor entries it "helps" cover
    # A candidate (U,V,W) contributes to entry (i,j,k) the value:
    # U[i*m+j] * V[j*p+k] * W[k*n+i]
    # We want the sum over selected candidates to equal 1 for each (i,j,k)
    
    # Build a matrix: rows = tensor entries (i,j,k), cols = candidates
    # M[r, c] = contribution of candidate c to entry r
    # We want to select a subset of columns such that sum of selected columns = ones vector
    
    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_entries = len(all_entries)
    num_candidates = len(candidates)
    
    # Build contribution matrix
    contrib = np.zeros((num_entries, num_candidates), dtype=float)
    for r, (i, j, k) in enumerate(all_entries):
        for c, (U, V, W) in enumerate(candidates):
            u_val = U.get(u_idx(i, j), 0)
            v_val = V.get(v_idx(j, k), 0)
            w_val = W.get(w_idx(k, i), 0)
            contrib[r, c] = u_val * v_val * w_val
    
    # Greedy set cover: repeatedly pick the candidate that covers the most uncovered entries
    # "Covering" an entry means contributing a non-zero value to it
    # But we need exact sum = 1, so this is more like a linear programming / basis pursuit
    
    # Better approach: use a greedy algorithm that tries to build up the solution
    # Start with empty solution, at each step pick the candidate that best reduces the residual
    
    target = np.ones(num_entries)
    residual = target.copy()
    selected = []
    
    # Precompute: for each candidate, which entries it affects
    # and its contribution pattern
    
    # Use a simpler greedy: at each step, find the candidate that minimizes ||residual - contrib[:,c]||
    # after adding it (with optimal sign/scaling if needed)
    
    # Actually, let's try a different approach: use integer linear programming or 
    # a smarter greedy that accounts for the fact we need exact 1s
    
    # For now, let's try the simple greedy and see if we can beat 54
    
    # Alternative: try all possible partitionings of the 5x6 grid into 2x2 blocks and leftovers
    # and compute the rank for each
    
    # Let me try a local search starting from the current solution
    
    # First, let's just try to verify the current approach works and then try to improve
    
    # Build the current solution as a starting point
    def build_current_solution():
        U_mat = np.zeros((n * m, 54))
        V_mat = np.zeros((m * p, 54))
        W_mat = np.zeros((p * n, 54))
        
        col = 0
        def add_rank_step(u_pairs, v_pairs, w_pairs):
            nonlocal col
            for idx, val in u_pairs:
                U_mat[idx, col] = val
            for idx, val in v_pairs:
                V_mat[idx, col] = val
            for idx, val in w_pairs:
                W_mat[idx, col] = val
            col += 1
        
        def add_strassen(j1, j2, k1, k2):
            add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1)], 
                         [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
            add_rank_step([(u_idx(0,j1), 1)], 
                         [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                         [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j2), 1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                         [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
            add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                         [(v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                         [(w_idx(k2,1), 1)])
            add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1)])
        
        def add_schoolbook_2x1x2(j, k1, k2):
            add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
            add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
            add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
            add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])
        
        add_strassen(0, 1, 0, 1)
        add_strassen(0, 1, 2, 3)
        add_strassen(0, 1, 4, 5)
        add_strassen(2, 3, 0, 1)
        add_schoolbook_2x1x2(4, 0, 1)
        add_strassen(2, 3, 2, 3)
        add_schoolbook_2x1x2(4, 2, 3)
        add_strassen(2, 3, 4, 5)
        add_schoolbook_2x1x2(4, 4, 5)
        
        return U_mat, V_mat, W_mat
    
    U_cur, V_cur, W_cur = build_current_solution()
    
    # Verify
    def verify(U, V, W, rank):
        # Check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        # T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_cur, V_cur, W_cur, 54):
        # Fallback to schoolbook
        U_list = [[0] * 60 for _ in range(n * m)]
        V_list = [[0] * 60 for _ in range(m * p)]
        W_list = [[0] * 60 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1
                    V_list[j*p+k][col] = 1
                    W_list[k*n+i][col] = 1
                    col += 1
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Now try to improve via local search
    # Try removing one term at a time and see if we can compensate with another
    # Or try replacing pairs of terms
    
    best_rank = 54
    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    # Try: for each pair of columns (r1, r2), try to replace them with a single column
    # that achieves the same effect (if possible)
    
    # This is complex. Let me try a different approach:
    # Use the candidate pool and do a greedy selection with exact arithmetic
    
    # Build contribution matrix more carefully
    # For greedy, at each step pick the candidate that maximizes the number of 
    # entries where it can help reduce the residual
    
    # Actually, let me try a simpler improvement: try different block partitionings
    
    # The 5x6 grid can be partitioned in different ways into 2x2 blocks and single rows/cols
    # Let me enumerate a few promising ones
    
    partitionings = []
    
    # Current: j={0,1},{2,3},{4}; k={0,1},{2,3},{4,5}
    partitionings.append(([0,1],[2,3],[4], [0,1],[2,3],[4,5]))
    
    # Try: j={0,1},{2,3},{4}; k={0,1},{2,3},{4,5} but different Strassen choices
    # Or: j={0,1},{2,3,4} with k split differently
    
    # Let me try: j={0,1},{2,3},{4} and k={0,1},{2,3,4,5}
    # For j={2,3,4} x k={2,3,4,5}: split as j={2,3}x k={2,3} (Strassen) + j={2,3}x k={4,5} (Strassen) + j={4}x k={2,3,4,5} (schoolbook 8)
    # That's 7+7+8 = 22 for that sub-block, vs current 7+4+7+4=22. Same.
    
    # Try: j={0,1,2} x k={0,1,2,3} + j={3,4} x k={0,1,2,3,4,5}
    # j={0,1,2}x k={0,1,2,3}: j={0,1}x k={0,1} (7) + j={0,1}x k={2,3} (7) + j={2}x k={0,1,2,3} (8) = 22
    # j={3,4}x k={0,1,2,3,4,5}: j={3,4}x k={0,1} (7) + j={3,4}x k={2,3} (7) + j={3,4}x k={4,5} (7) = 21
    # Total: 43. Better than 54!
    
    # Let me verify this partitioning works
    def build_solution_j012_k0123_j34_k012345():
        U_mat = np.zeros((n * m, 43))
        V_mat = np.zeros((m * p, 43))
        W_mat = np.zeros((p * n, 43))
        
        col = 0
        def add_rank_step(u_pairs, v_pairs, w_pairs):
            nonlocal col
            if col >= 43:
                return False
            for idx, val in u_pairs:
                U_mat[idx, col] = val
            for idx, val in v_pairs:
                V_mat[idx, col] = val
            for idx, val in w_pairs:
                W_mat[idx, col] = val
            col += 1
            return True
        
        def add_strassen(j1, j2, k1, k2):
            add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1)], 
                         [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
            add_rank_step([(u_idx(0,j1), 1)], 
                         [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                         [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j2), 1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                         [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
            add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                         [(v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                         [(w_idx(k2,1), 1)])
            add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1)])
        
        def add_schoolbook_2x1xK(j, k_list):
            for k in k_list:
                add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,0), 1)])
                add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,1), 1)])
        
        # j={0,1,2} x k={0,1,2,3}
        add_strassen(0, 1, 0, 1)  # 7
        add_strassen(0, 1, 2, 3)  # 7
        add_schoolbook_2x1xK(2, [0, 1, 2, 3])  # 8
        
        # j={3,4} x k={0,1,2,3,4,5}
        add_strassen(3, 4, 0, 1)  # 7
        add_strassen(3, 4, 2, 3)  # 7
        add_strassen(3, 4, 4, 5)  # 7
        
        return U_mat, V_mat, W_mat, col
    
    U2, V2, W2, rank2 = build_solution_j012_k0123_j34_k012345()
    if verify(U2, V2, W2, rank2):
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2
    
    # Try more partitionings
    # j={0,1,2,3} x k={0,1,2,3} + j={4} x k={0,1,2,3,4,5}
    # j={0,1,2,3}x k={0,1,2,3}: j={0,1}x k={0,1}(7) + j={0,1}x k={2,3}(7) + j={2,3}x k={0,1}(7) + j={2,3}x k={2,3}(7) = 28
    # j={4}x k={0,1,2,3,4,5}: 12
    # Total: 40
    
    def build_solution_j0123_k0123_j4_k012345():
        U_mat = np.zeros((n * m, 40))
        V_mat = np.zeros((m * p, 40))
        W_mat = np.zeros((p * n, 40))
        
        col = 0
        def add_rank_step(u_pairs, v_pairs, w_pairs):
            nonlocal col
            if col >= 40:
                return False
            for idx, val in u_pairs:
                U_mat[idx, col] = val
            for idx, val in v_pairs:
                V_mat[idx, col] = val
            for idx, val in w_pairs:
                W_mat[idx, col] = val
            col += 1
            return True
        
        def add_strassen(j1, j2, k1, k2):
            add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1)], 
                         [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
            add_rank_step([(u_idx(0,j1), 1)], 
                         [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                         [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j2), 1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                         [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
            add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                         [(v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                         [(w_idx(k2,1), 1)])
            add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1)])
        
        def add_schoolbook_2x1xK(j, k_list):
            for k in k_list:
                add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,0), 1)])
                add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,1), 1)])
        
        # j={0,1,2,3} x k={0,1,2,3}
        add_strassen(0, 1, 0, 1)
        add_strassen(0, 1, 2, 3)
        add_strassen(2, 3, 0, 1)
        add_strassen(2, 3, 2, 3)
        
        # j={4} x k={0,1,2,3,4,5}
        add_schoolbook_2x1xK(4, [0, 1, 2, 3, 4, 5])
        
        return U_mat, V_mat, W_mat, col
    
    U3, V3, W3, rank3 = build_solution_j0123_k0123_j4_k012345()
    if verify(U3, V3, W3, rank3):
        if rank3 < best_rank:
            best_rank = rank3
            best_U, best_V, best_W = U3, V3, W3
    
    # Try: j={0,1,2,3} x k={0,1,2,3,4,5} + j={4} x k={0,1,2,3,4,5}
    # j={0,1,2,3}x k={0,1,2,3,4,5}: 4 Strassen blocks x 7 = 28
    # j={4}x k={0,1,2,3,4,5}: 12
    # Total: 40
    
    def build_solution_j0123_k012345_j4_k012345():
        U_mat = np.zeros((n * m, 40))
        V_mat = np.zeros((m * p, 40))
        W_mat = np.zeros((p * n, 40))
        
        col = 0
        def add_rank_step(u_pairs, v_pairs, w_pairs):
            nonlocal col
            if col >= 40:
                return False
            for idx, val in u_pairs:
                U_mat[idx, col] = val
            for idx, val in v_pairs:
                V_mat[idx, col] = val
            for idx, val in w_pairs:
                W_mat[idx, col] = val
            col += 1
            return True
        
        def add_strassen(j1, j2, k1, k2):
            add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                         [(v_idx(j1,k1), 1)], 
                         [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
            add_rank_step([(u_idx(0,j1), 1)], 
                         [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                         [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
            add_rank_step([(u_idx(1,j2), 1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                         [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
            add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                         [(v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
            add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                         [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                         [(w_idx(k2,1), 1)])
            add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                         [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                         [(w_idx(k1,0), 1)])
        
        def add_schoolbook_2x1xK(j, k_list):
            for k in k_list:
                add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,0), 1)])
                add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,1), 1)])
        
        # j={0,1,2,3} x k={0,1,2,3,4,5}
        add_strassen(0, 1, 0, 1)
        add_strassen(0, 1, 2, 3)
        add_strassen(0, 1, 4, 5)
        add_strassen(2, 3, 0, 1)
        add_strassen(2, 3, 2, 3)
        add_strassen(2, 3, 4, 5)
        
        # j={4} x k={0,1,2,3,4,5}
        add_schoolbook_2x1xK(4, [0, 1, 2, 3, 4, 5])
        
        return U_mat, V_mat, W_mat, col
    
    U4, V4, W4, rank4 = build_solution_j0123_k012345_j4_k012345()
    if verify(U4, V4, W4, rank4):
        if rank4 < best_rank:
            best_rank = rank4
            best_U, best_V, best_W = U4, V4, W4
    
    # Try: j={0,1,2} x k={0,1,2,3,4,5} + j={3,4} x k={0,1,2,3,4,5}
    # j={0,1,2}x k={0,1,2,3,4,5}: j={0,1}x k={0,1}(7) + j={0,1}x k={2,3}(7) + j={0,1}x k={4,5}(7) + j={2}x k={0,1,2,3,4,5}(12) = 33
    # j={3,4}x k={0,1,2,3,4,5}: 3 Strassen x 7 = 21
    # Total: 54. Same as current.
    
    # Try: j={0,1} x k={0,1,2,3,4,5} + j={2,3,4} x k={0,1,2,3,4,5}
    # j={0,1}x k={0,1,2,3,4,5}: 3 Strassen x 7 = 21
    # j={2,3,4}x k={0,1,2,3,4,5}: j={2,3}x k={0,1}(7) + j={2,3}x k={2,3}(7) + j={2,3}x k={4,5}(7) + j={4}x k={0,1,2,3,4,5}(12) = 33
    # Total: 54. Same.
    
    # The best so far is 40. Let me try to see if we can do even better.
    # j={0,1,2,3,4} x k={0,1,2,3,4,5}
    # = j={0,1,2,3}x k={0,1,2,3,4,5} (28) + j={4}x k={0,1,2,3,4,5} (12) = 40
    # Or = j={0,1,2}x k={0,1,2,3,4,5} (33) + j={3,4}x k={0,1,2,3,4,5} (21) = 54
    # Or = j={0,1}x k={0,1,2,3,4,5} (21) + j={2,3,4}x k={0,1,2,3,4,5} (33) = 54
    
    # What about non-rectangular partitions?
    # j={0,1,2,3} x k={0,1,2,3} (28) + j={0,1,2,3} x k={4,5} (14) + j={4} x k={0,1,2,3,4,5} (12) = 54. Worse.
    
    # j={0,1,2,3} x k={0,1,2,3} (28) + j={4} x k={0,1,2,3,4,5} (12) = 40. Same as before.
    
    # Can we do better than 40? The theoretical lower bound is... 
    # For 2x5x6, the rank is at least max(2*5, 5*6, 2*6) = 30, but that's a weak bound.
    # Actually, the rank of matrix multiplication tensor M_{n,m,p} is at least n*m*p / something...
    # The known lower bounds are not tight for small sizes.
    
    # Let me try a few more creative partitionings
    
    # j={0,1,2,3,4} x k={0,1,2,3,4,5}
    # Try: j={0,1}x k={0,1,2,3} (14) + j={0,1}x k={4,5} (7) + j={2,3}x k={0,1,2,3} (14) + j={2,3}x k={4,5} (7) + j={4}x k={0,1,2,3,4,5} (12) = 54
    
    # Try: j={0,1,2}x k={0,1,2} (7+7+4=18) + j={0,1,2}x k={3,4,5} (7+7+4=18) + j={3,4}x k={0,1,2,3,4,5} (21) = 57. Worse.
    
    # I think 40 might be the best with this simple partitioning approach.
    # Let me try to use the greedy set cover to see if we can do better.
    
    if time.time() - start_time < deadline - 10:
        # Try greedy set cover on the candidate pool
        # Build contribution matrix
        all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        num_entries = len(all_entries)
        
        # Regenerate candidates more carefully
        candidates2 = []
        
        # Schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = {u_idx(i, j): 1}
                    V = {v_idx(j, k): 1}
                    W = {w_idx(k, i): 1}
                    candidates2.append((U, V, W))
        
        # Strassen terms for all 2x2x2 blocks
        for j1 in range(m - 1):
            j2 = j1 + 1
            for k1 in range(p - 1):
                k2 = k1 + 1
                candidates2.append(
                    ({u_idx(0,j1): 1, u_idx(1,j2): 1}, 
                     {v_idx(j1,k1): 1, v_idx(j2,k2): 1}, 
                     {w_idx(k1,0): 1, w_idx(k2,1): 1})
                )
                candidates2.append(
                    ({u_idx(1,j1): 1, u_idx(1,j2): 1}, 
                     {v_idx(j1,k1): 1}, 
                     {w_idx(k1,1): 1, w_idx(k2,1): -1})
                )
                candidates2.append(
                    ({u_idx(0,j1): 1}, 
                     {v_idx(j1,k2): 1, v_idx(j2,k2): -1}, 
                     {w_idx(k2,0): 1, w_idx(k2,1): 1})
                )
                candidates2.append(
                    ({u_idx(1,j2): 1}, 
                     {v_idx(j2,k1): 1, v_idx(j1,k1): -1}, 
                     {w_idx(k1,0): 1, w_idx(k1,1): 1})
                )
                candidates2.append(
                    ({u_idx(0,j1): 1, u_idx(0,j2): 1}, 
                     {v_idx(j2,k2): 1}, 
                     {w_idx(k1,0): -1, w_idx(k2,0): 1})
                )
                candidates2.append(
                    ({u_idx(1,j1): 1, u_idx(0,j1): -1}, 
                     {v_idx(j1,k1): 1, v_idx(j1,k2): 1}, 
                     {w_idx(k2,1): 1})
                )
                candidates2.append(
                    ({u_idx(0,j2): 1, u_idx(1,j2): -1}, 
                     {v_idx(j2,k1): 1, v_idx(j2,k2): 1}, 
                     {w_idx(k1,0): 1})
                )
        
        num_cands = len(candidates2)
        contrib2 = np.zeros((num_entries, num_cands), dtype=float)
        for r, (i, j, k) in enumerate(all_entries):
            for c, (U, V, W) in enumerate(candidates2):
                u_val = U.get(u_idx(i, j), 0)
                v_val = V.get(v_idx(j, k), 0)
                w_val = W.get(w_idx(k, i), 0)
                contrib2[r, c] = u_val * v_val * w_val
        
        # Greedy: pick candidates that help reduce residual
        residual = np.ones(num_entries)
        selected2 = []
        
        for step in range(60):
            if time.time() - start_time > deadline - 5:
                break
            if np.allclose(residual, 0, atol=1e-9):
                break
            
            best_c = -1
            best_score = -1e18
            for c in range(num_cands):
                if c in selected2:
                    continue
                # Score: how well does adding this candidate reduce ||residual||?
                new_residual = residual - contrib2[:, c]
                score = -np.sum(new_residual ** 2)
                if score > best_score:
                    best_score = score
                    best_c = c
            
            if best_c < 0:
                break
            selected2.append(best_c)
            residual -= contrib2[:, best_c]
        
        if len(selected2) < best_rank and np.allclose(residual, 0, atol=1e-9):
            # Build the solution
            U_g = np.zeros((n * m, len(selected2)))
            V_g = np.zeros((m * p, len(selected2)))
            W_g = np.zeros((p * n, len(selected2)))
            for r, c in enumerate(selected2):
                U_dict, V_dict, W_dict = candidates2[c]
                for idx, val in U_dict.items():
                    U_g[idx, r] = val
                for idx, val in V_dict.items():
                    V_g[idx, r] = val
                for idx, val in W_dict.items():
                    W_g[idx, r] = val
            
            if verify(U_g, V_g, W_g, len(selected2)):
                best_rank = len(selected2)
                best_U, best_V, best_W = U_g, V_g, W_g
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
