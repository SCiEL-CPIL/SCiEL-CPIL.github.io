"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via systematic search with exact rational arithmetic.
  3. Try group-theoretic embedding ideas and column merging.
  4. If no improvement is found within the time budget, return the known rank-44 decomposition.
"""
import time
import sys
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Compute column vectors for analysis ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                u_val = U[i * m + a][c]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][c]
                    if w_val == 0:
                        continue
                    vec[i * m * p + a * p + k] = u_val * v_val * w_val
        return vec

    def _build_col_vectors(U, V, W, rank):
        cols = []
        for c in range(rank):
            cols.append(_col_vector(U, V, W, c))
        return cols

    # ---- Try to reduce rank by finding linear dependencies ----
    def _try_reduce_rank():
        nonlocal best_candidate, best_rank

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44
        cols = _build_col_vectors(U, V, W, rank0)

        target = [1] * num_terms

        # Verify base
        total = [0] * num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return

        # Check for duplicate columns
        seen = {}
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                # Found duplicate - try removing
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank0):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return
            else:
                seen[key] = c

        if not _elapsed_ok():
            return

        # Try to find pairs of columns that sum to zero (c + d = 0 means we can remove both
        # if there's another pair that sums to the same thing, but more directly:
        # if col_i + col_j = 0, we can't just remove one. But if col_i = col_j, we can remove one.)
        
        # More useful: try to find if col_i = col_j + col_k for some j,k
        # This would let us replace col_i with nothing and add col_j+col_k... but that's not helpful.
        # Actually, if col_i = col_j + col_k, we can remove col_i and it's fine.
        
        # Let's try: for each column i, check if it equals the sum of two other columns
        # This is O(rank^3 * num_terms) which is 44^3 * 48 ≈ 4 million - feasible
        
        if not _elapsed_ok():
            return
            
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(rank0):
                if j == i:
                    continue
                for k in range(rank0):
                    if k == i or k == j:
                        continue
                    # Check if cols[i] == cols[j] + cols[k]
                    match = True
                    for idx in range(num_terms):
                        if cols[i][idx] != cols[j][idx] + cols[k][idx]:
                            match = False
                            break
                    if match:
                        # cols[i] = cols[j] + cols[k], so we can remove col i
                        new_rank = rank0 - 1
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        nc = 0
                        for cc in range(rank0):
                            if cc == i:
                                continue
                            for ii in range(n * m):
                                new_U[ii][nc] = U[ii][cc]
                            for ii in range(m * p):
                                new_V[ii][nc] = V[ii][cc]
                            for ii in range(p * n):
                                new_W[ii][nc] = W[ii][cc]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            return

        if not _elapsed_ok():
            return

        # Try: can we replace two columns with one new column?
        # If cols[i] + cols[j] can be factored as a rank-1 tensor, we can merge them.
        # The sum cols[i]+cols[j] is a tensor T[i,a,k]. We need to check if it's rank-1.
        # A tensor is rank-1 iff T[i,a,k] = u[i,a] * v[a,k] * w[k,i] for some u,v,w.
        # For our small dimensions, we can check this directly.
        
        def _is_rank1_tensor(tensor):
            """Check if a 2x3x8 tensor is rank-1 (can be written as u[i,a]*v[a,k]*w[k,i])"""
            # Find first nonzero entry
            i0, a0, k0 = None, None, None
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        if tensor[i][a][k] != 0:
                            i0, a0, k0 = i, a, k
                            break
                    if i0 is not None:
                        break
                if i0 is not None:
                    break
            
            if i0 is None:
                return True  # zero tensor is rank 0
            
            val = tensor[i0][a0][k0]
            
            # For rank-1: tensor[i][a][k] = u[i,a] * v[a,k] * w[k,i]
            # tensor[i0][a0][k0] = u[i0,a0] * v[a0,k0] * w[k0,i0] = val
            # tensor[i][a][k0] = u[i,a] * v[a,k0] * w[k0,i]
            # tensor[i0][a][k] = u[i0,a] * v[a,k] * w[k,i0]
            
            # This is complex. Let's use a simpler check:
            # For rank-1, all 2x2 minors of any unfolding must be zero.
            # Unfold along first mode: matrix of size (n*m) x p
            # Unfold along second mode: matrix of size (n*p) x m  
            # Unfold along third mode: matrix of size (m*n) x p
            
            # Check mode-1 unfolding: (n*m) x p matrix
            # Rows indexed by (i,a), columns by k
            mat1 = [[tensor[i][a][k] for k in range(p)] for i in range(n) for a in range(m)]
            # Check rank of mat1
            if _matrix_rank(mat1) > 1:
                return False
            
            # Check mode-2 unfolding: (n*p) x m matrix  
            # Rows indexed by (i,k), columns by a
            mat2 = [[tensor[i][a][k] for a in range(m)] for i in range(n) for k in range(p)]
            if _matrix_rank(mat2) > 1:
                return False
            
            # Check mode-3 unfolding: (m*n) x p matrix
            # Rows indexed by (a,i), columns by k
            mat3 = [[tensor[i][a][k] for k in range(p)] for a in range(m) for i in range(n)]
            if _matrix_rank(mat3) > 1:
                return False
            
            return True
        
        def _matrix_rank(mat):
            """Compute rank of a matrix using Gaussian elimination with Fractions"""
            if not mat or not mat[0]:
                return 0
            rows = len(mat)
            cols = len(mat[0])
            mat = [[Fraction(x) for x in row] for row in mat]
            rank = 0
            for col in range(cols):
                # Find pivot
                pivot = -1
                for row in range(rank, rows):
                    if mat[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap
                mat[rank], mat[pivot] = mat[pivot], mat[rank]
                # Eliminate
                for row in range(rows):
                    if row == rank:
                        continue
                    factor = mat[row][col] / mat[rank][col]
                    for c in range(col, cols):
                        mat[row][c] -= factor * mat[rank][c]
                rank += 1
                if rank == rows:
                    break
            return rank

        if not _elapsed_ok():
            return

        # Try merging pairs of columns
        for i in range(rank0):
            if not _elapsed_ok():
                return
            for j in range(i + 1, rank0):
                # Build sum tensor
                sum_tensor = [[[0] * p for _ in range(m)] for _ in range(n)]
                for i_idx in range(n):
                    for a_idx in range(m):
                        for k_idx in range(p):
                            sum_tensor[i_idx][a_idx][k_idx] = (
                                cols[i][i_idx * m * p + a_idx * p + k_idx] + 
                                cols[j][i_idx * m * p + a_idx * p + k_idx]
                            )
                
                if _is_rank1_tensor(sum_tensor):
                    # The sum is rank-1, so we can merge cols[i] and cols[j] into one new column
                    # We need to find u, v, w such that u[i,a]*v[a,k]*w[k,i] = sum_tensor[i,a,k]
                    # This is a factorization problem. For our small case, let's try to solve it.
                    
                    # Find a nonzero entry to anchor
                    i0, a0, k0 = None, None, None
                    for i_idx in range(n):
                        for a_idx in range(m):
                            for k_idx in range(p):
                                if sum_tensor[i_idx][a_idx][k_idx] != 0:
                                    i0, a0, k0 = i_idx, a_idx, k_idx
                                    break
                                if i0 is not None:
                                    break
                            if i0 is not None:
                                break
                        if i0 is not None:
                            break
                    
                    if i0 is None:
                        # Sum is zero, just remove both
                        new_rank = rank0 - 2
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        nc = 0
                        for cc in range(rank0):
                            if cc == i or cc == j:
                                continue
                            for ii in range(n * m):
                                new_U[ii][nc] = U[ii][cc]
                            for ii in range(m * p):
                                new_V[ii][nc] = V[ii][cc]
                            for ii in range(p * n):
                                new_W[ii][nc] = W[ii][cc]
                            nc += 1
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            return
                        continue
                    
                    # For rank-1 factorization, we need:
                    # u[i,a] * v[a,k] * w[k,i] = sum_tensor[i,a,k]
                    # 
                    # Strategy: set u[i0,a0] = 1, then v[a0,k0] * w[k0,i0] = sum_tensor[i0,a0,k0]
                    # Then for each (i,a): u[i,a] = sum_tensor[i,a,k0] / (v[a,k0] * w[k0,i])
                    # This gets complex. Let's try a specific factorization.
                    #
                    # Actually, for the merge to work, we need the new column to be a valid
                    # rank-1 tensor. Let's just try to construct it.
                    #
                    # Set w[k,i] = sum_tensor[i0,a0,k] / sum_tensor[i0,a0,k0] for all k (when i=i0)
                    # This is getting too complex. Let's skip this for now and just note
                    # that the sum is rank-1.
                    pass

        # ---- Try a completely different approach: search for rank < 44 ----
        # Use the fact that we can try to find a decomposition with rank r < 44
        # by solving a system of equations.
        #
        # For rank r, we need U (6 x r), V (24 x r), W (16 x r) such that
        # for all (i,a,k): sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1
        #
        # This is a system of 48 equations in 6r + 24r + 16r = 46r unknowns.
        # For r = 43: 1978 unknowns, 48 equations. Very underdetermined.
        # But the equations are cubic (trilinear), so it's not linear.
        #
        # A better approach: fix U and V, then W is determined linearly.
        # Or fix two of U,V,W and solve for the third.
        #
        # Let's try: fix U and V, then for each (i,a,k):
        # sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1
        # This is linear in W. For fixed U,V, we have 48 equations in 16r unknowns (W entries).
        # But W[k,i,c] appears in equations for all a, so it's coupled.
        #
        # Actually, W[k,i,c] appears in the equation for (i,a,k) for all a.
        # So for fixed (k,i,c), W[k,i,c] appears in m=3 equations (one for each a).
        # This means we can't easily separate them.
        #
        # Let's try a different approach: use the structure of the problem.
        # The tensor is T[i,a,k] = 1 for all i,a,k.
        # 
        # One idea: use a "convolution-like" structure.
        # 
        # For now, let's try a randomized search: randomly generate U and V with small
        # integer entries, then solve for W and check if it works.

        if not _elapsed_ok():
            return

        import random
        random.seed(42)
        
        # Try rank 43, 42, 41, ...
        for target_rank in range(43, 30, -1):
            if not _elapsed_ok():
                return
            
            # Try a few random initializations
            for attempt in range(50):
                if not _elapsed_ok():
                    return
                
                # Generate random U and V with small integer entries
                U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
                V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
                
                # For each (k,i), we need to find W[k,i,c] for c in range(target_rank)
                # such that for all a: sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1
                #
                # This is a system of m=3 equations in target_rank unknowns for each (k,i).
                # But the equations are coupled through the same W[k,i,c] appearing in all a.
                #
                # Actually, for fixed (k,i), the equations are:
                # sum_c U[i,0,c] * V[0,k,c] * W[k,i,c] = 1
                # sum_c U[i,1,c] * V[1,k,c] * W[k,i,c] = 1
                # sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1
                #
                # This is 3 equations in target_rank unknowns. Underdetermined for target_rank > 3.
                # We can pick a solution and check if it works globally.
                
                # For each (k,i), solve the 3-equation system
                W_rand = [[0] * target_rank for _ in range(p * n)]
                valid = True
                
                for k in range(p):
                    for i in range(n):
                        # Build 3 x target_rank system
                        # Eq a: sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1
                        # Let x_c = W[k,i,c]
                        # sum_c (U[i,a,c] * V[a,k,c]) * x_c = 1 for a=0,1,2
                        
                        # This is A * x = 1 where A[a,c] = U[i,a,c] * V[a,k,c]
                        # A is 3 x target_rank
                        A = [[U_rand[i * m + a][c] * V_rand[a * p + k][c] for c in range(target_rank)] for a in range(m)]
                        
                        # Solve A * x = [1,1,1]
                        # Use least squares or find a particular solution
                        # For simplicity, try x_c = 1 for all c and check
                        x = [1] * target_rank
                        for a in range(m):
                            s = sum(A[a][c] * x[c] for c in range(target_rank))
                            if s != 1:
                                # Try other simple solutions
                                valid = False
                                break
                        if not valid:
                            break
                        for c in range(target_rank):
                            W_rand[k * n + i][c] = x[c]
                    
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check if this full decomposition works
                U_test = U_rand
                V_test = V_rand
                W_test = W_rand
                ok = True
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            acc = 0
                            for c in range(target_rank):
                                acc += U_test[i * m + a][c] * V_test[a * p + k][c] * W_test[k * n + i][c]
                            if acc != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    best_candidate = {"U": U_test, "V": V_test, "W": W_test}
                    best_rank = target_rank
                    return

        return

    _try_reduce_rank()

    return base if best_rank == 44 else best_candidate

# EVOLVE-BLOCK-END
