# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

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

    # ---- Try to reduce rank via linear dependency elimination ----
    def _try_reduce():
        nonlocal best_candidate, best_rank

        num_terms = n * m * p  # 48

        def _get_term_vector(U, V, W, c):
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

        U = best_candidate["U"]
        V = best_candidate["V"]
        W = best_candidate["W"]
        rank0 = len(U[0])

        # Build matrix of term vectors (rows = tensor entries, cols = terms)
        # M[idx, c] = term_c[idx]
        M = [[0] * rank0 for _ in range(num_terms)]
        for c in range(rank0):
            vec = _get_term_vector(U, V, W, c)
            for idx in range(num_terms):
                M[idx][c] = vec[idx]

        # We look for integer linear dependencies: sum alpha_c * col_c = 0
        # We use Gaussian elimination over rationals (or integers via Bareiss) to find null space vectors.
        # Since rank is small (44), we can do this.
        
        # To find one null vector, we can perform Gaussian elimination on M^T (or M) 
        # and look for free variables.
        
        # Let's use a simple approach: try to eliminate one column at a time if it is a 
        # linear combination of the others with integer coefficients.
        
        # We will iterate through columns and check if col_k can be expressed as 
        # integer combination of other columns.
        
        # To do this efficiently, we can maintain the row echelon form of the matrix 
        # excluding column k, and check if col_k is in its span with integer coeffs.
        
        # However, doing this for all k is O(R^4). R=44, 44^4 is ~3.7M. Feasible.
        
        # Optimization: Use NumPy for floating point checks first? 
        # No, we need exact integers.
        
        # Let's try a greedy approach:
        # For each column k, try to express it as a combination of previous columns (0..k-1).
        # If we can, we might be able to swap/eliminate? 
        # Actually, if col_k = sum_{j < k} alpha_j col_j, then 
        # T = sum_{j < k} col_j + col_k + sum_{j > k} col_j
        #   = sum_{j < k} (1 + alpha_j) col_j + sum_{j > k} col_j
        # This eliminates col_k.
        # We need alpha_j to be integers so that (1+alpha_j) are integers.
        
        # Algorithm:
        # 1. Build M.
        # 2. Iterate k from 0 to R-1.
        # 3. Check if M[:, k] is in the integer span of M[:, 0..k-1] (or any subset?).
        #    Using only previous columns is a heuristic that preserves the "order" but might miss deps.
        #    A better heuristic: Check if M[:, k] is in the integer span of ALL OTHER columns.
        #    This is expensive.
        
        # Let's stick to the "previous columns" heuristic for speed, or "all columns" if time permits.
        # Given the structure, dependencies might be global.
        
        # Let's try to find ANY integer dependency vector alpha such that 
        # sum alpha_c col_c = 0.
        # Then for any k with alpha_k != 0, we can eliminate k if 
        # alpha_j / alpha_k is an integer for all j != k.
        
        # Finding one dependency vector:
        # Perform Gaussian elimination on M (num_terms x rank0) over Q.
        # If rank(M) < rank0, there is a null vector.
        
        # We'll compute the null space basis.
        
        # Use Fractions for exactness.
        mat = [row[:] for row in M]
        rows = num_terms
        cols = rank0
        
        # Gaussian elimination to find pivot columns
        pivot_cols = []
        current_row = 0
        for c in range(cols):
            # Find pivot in rows current_row..rows-1
            pivot_row = -1
            for r in range(current_row, rows):
                if mat[r][c] != 0:
                    pivot_row = r
                    break
            
            if pivot_row == -1:
                continue
                
            # Swap
            mat[current_row], mat[pivot_row] = mat[pivot_row], mat[current_row]
            pivot_val = mat[current_row][c]
            
            # Normalize pivot row to make pivot 1 (using Fraction)
            # Actually, we can just eliminate below
            for r in range(current_row + 1, rows):
                if mat[r][c] != 0:
                    factor = Fraction(mat[r][c]) / Fraction(pivot_val)
                    for c2 in range(c, cols):
                        mat[r][c2] -= factor * mat[current_row][c2]
            
            pivot_cols.append(c)
            current_row += 1
            
            if current_row == rows:
                break
        
        matrix_rank = len(pivot_cols)
        
        if matrix_rank == cols:
            # No dependencies
            return
        
        # There are free columns.
        # The null space dimension is cols - matrix_rank.
        # We can construct a null vector by setting free variables to 1 and solving for pivots.
        
        # Let's find one null vector.
        # We need to back-substitute.
        # The matrix is in row echelon form (not reduced).
        
        # Let's store the reduced row echelon form (RREF) to easily find null vectors.
        # Re-do elimination to RREF?
        # Or just use the fact that we have pivot_cols.
        
        # Let's identify free columns
        free_cols = [c for c in range(cols) if c not in pivot_cols]
        
        # Construct a null vector for the first free column
        # Set alpha[free_col] = 1, other free = 0.
        # Solve for pivot variables.
        
        alpha = [0] * cols
        # We need the RREF to do this easily.
        # Let's perform back substitution on the echelon form we have.
        # The matrix `mat` is now in row echelon form.
        # mat[pivot_idx][pivot_col] is the pivot.
        
        # Let's create a map from pivot_col to row index
        # Actually, the rows in `mat` 0..matrix_rank-1 correspond to pivot_cols.
        
        # Let's set the free variable
        target_free = free_cols[0]
        alpha[target_free] = 1
        
        # We need to solve:
        # sum_c mat[r][c] alpha[c] = 0 for r in 0..matrix_rank-1
        # Note: mat is in REF.
        # For each pivot row r (from bottom to top):
        #   mat[r][pivot_col_r] * alpha[pivot_col_r] + sum_{c > pivot_col_r} mat[r][c] alpha[c] = 0
        #   => alpha[pivot_col_r] = - sum_{c > pivot_col_r} mat[r][c] alpha[c] / mat[r][pivot_col_r]
        
        # We iterate r from matrix_rank-1 down to 0.
        for r in range(matrix_rank - 1, -1, -1):
            pc = pivot_cols[r]
            sum_val = 0
            for c in range(pc + 1, cols):
                if alpha[c] != 0 and mat[r][c] != 0:
                    sum_val += mat[r][c] * alpha[c]
            
            # alpha[pc] = -sum_val / mat[r][pc]
            # We need this to be an integer (or rational that allows integer scaling later)
            # Ideally, we want alpha to be integers.
            # If mat[r][pc] divides sum_val, alpha[pc] is integer.
            
            num = -sum_val
            den = mat[r][pc]
            
            # Check divisibility
            if num % den != 0:
                # Not an integer solution for this specific free variable assignment?
                # We might need to scale the whole vector later.
                # Let's store as Fraction for now
                alpha[pc] = Fraction(num, den)
            else:
                alpha[pc] = num // den
        
        # Now alpha is a vector in the null space (over Q).
        # We want to eliminate a term k.
        # Condition: alpha_j / alpha_k is an integer for all j != k.
        
        # Find a k such that alpha_k != 0 and alpha_j is divisible by alpha_k (in Q sense, 
        # i.e., alpha_j / alpha_k is integer) for all j.
        
        # To make it easier, let's find the LCM of denominators of alpha and scale alpha to integers.
        
        # Convert alpha to Fractions
        alpha_frac = [Fraction(x) for x in alpha]
        
        # Find LCM of denominators
        from math import gcd
        def lcm(a, b):
            return abs(a * b) // gcd(a, b) if a and b else 0
        
        l = 1
        for x in alpha_frac:
            l = lcm(l, x.denominator)
            
        # Scale
        alpha_int = [int(x * l) for x in alpha_frac]
        
        # Now check for elimination
        # Find k such that alpha_int[k] != 0
        # And for all j != k, alpha_int[j] % alpha_int[k] == 0 ?
        # No, the condition was: new_coeff_j = 1 + alpha_j / (-alpha_k) is integer.
        # So alpha_j / alpha_k must be integer.
        
        found_elimination = False
        for k in range(cols):
            if alpha_int[k] == 0:
                continue
            
            valid = True
            for j in range(cols):
                if j == k:
                    continue
                if alpha_int[j] % alpha_int[k] != 0:
                    valid = False
                    break
            
            if valid:
                # We can eliminate k
                # New coefficients for j: 1 - alpha_int[j] / alpha_int[k]
                # Let's construct the new U, V, W
                
                new_rank = cols - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                nc = 0
                for j in range(cols):
                    if j == k:
                        continue
                    
                    # Scale factor
                    # Original term j has coeff 1.
                    # We add (alpha_j / -alpha_k) * term_k? 
                    # Wait. 
                    # T = sum term_j.
                    # Dep: sum alpha_j term_j = 0 => term_k = - sum_{j!=k} (alpha_j/alpha_k) term_j.
                    # T = term_k + sum_{j!=k} term_j = sum_{j!=k} (1 - alpha_j/alpha_k) term_j.
                    # So new coeff for j is (1 - alpha_j/alpha_k).
                    
                    coeff = 1 - (alpha_int[j] // alpha_int[k])
                    
                    # Scale U[:, j] by coeff
                    for i in range(n * m):
                        new_U[i][nc] = U[i][j] * coeff
                    for i in range(m * p):
                        new_V[i][nc] = V[i][j]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][j]
                        
                    nc += 1
                
                cand = {"U": new_U, "V": new_V, "W": new_W}
                
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    found_elimination = True
                    # Break and restart search with new candidate?
                    # Or just return?
                    # Restarting is better to find more reductions.
                    break
        
        if found_elimination:
            # Recursive call or loop?
            # To avoid deep recursion, we can just return and let the caller loop?
            # But construct() calls _try_reduce() once.
            # Let's make _try_reduce loop until no more improvements.
            pass

    # Modify _try_reduce to loop
    # We need to wrap the logic in a loop.
    
    # Let's redefine _try_reduce to handle the loop internally.
    
    def _try_reduce_loop():
        nonlocal best_candidate, best_rank
        
        while _elapsed_ok():
            U = best_candidate["U"]
            V = best_candidate["V"]
            W = best_candidate["W"]
            rank0 = len(U[0])
            
            if rank0 <= 1:
                break
                
            num_terms = n * m * p
            
            def _get_term_vector(U, V, W, c):
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

            M = [[0] * rank0 for _ in range(num_terms)]
            for c in range(rank0):
                vec = _get_term_vector(U, V, W, c)
                for idx in range(num_terms):
                    M[idx][c] = vec[idx]

            # Gaussian elimination
            mat = [row[:] for row in M]
            rows = num_terms
            cols = rank0
            pivot_cols = []
            current_row = 0
            for c in range(cols):
                pivot_row = -1
                for r in range(current_row, rows):
                    if mat[r][c] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                mat[current_row], mat[pivot_row] = mat[pivot_row], mat[current_row]
                pivot_val = mat[current_row][c]
                for r in range(current_row + 1, rows):
                    if mat[r][c] != 0:
                        factor = Fraction(mat[r][c]) / Fraction(pivot_val)
                        for c2 in range(c, cols):
                            mat[r][c2] -= factor * mat[current_row][c2]
                pivot_cols.append(c)
                current_row += 1
                if current_row == rows:
                    break
            
            matrix_rank = len(pivot_cols)
            
            if matrix_rank == cols:
                break # No dependencies
            
            free_cols = [c for c in range(cols) if c not in pivot_cols]
            
            # Try to find a valid elimination
            eliminated = False
            for target_free in free_cols:
                if not _elapsed_ok():
                    break
                
                alpha = [0] * cols
                alpha[target_free] = 1
                
                # Back substitution
                valid_int = True
                for r in range(matrix_rank - 1, -1, -1):
                    pc = pivot_cols[r]
                    sum_val = 0
                    for c in range(pc + 1, cols):
                        if alpha[c] != 0 and mat[r][c] != 0:
                            sum_val += mat[r][c] * alpha[c]
                    
                    num = -sum_val
                    den = mat[r][pc]
                    
                    if num % den != 0:
                        valid_int = False
                        break
                    alpha[pc] = num // den
                
                if not valid_int:
                    continue
                
                # Check elimination condition
                # Find k with alpha[k] != 0
                # such that alpha[j] % alpha[k] == 0 for all j
                for k in range(cols):
                    if alpha[k] == 0:
                        continue
                    
                    valid_elim = True
                    for j in range(cols):
                        if j == k:
                            continue
                        if alpha[j] % alpha[k] != 0:
                            valid_elim = False
                            break
                    
                    if valid_elim:
                        # Construct new candidate
                        new_rank = cols - 1
                        new_U = [[0] * new_rank for _ in range(n * m)]
                        new_V = [[0] * new_rank for _ in range(m * p)]
                        new_W = [[0] * new_rank for _ in range(p * n)]
                        
                        nc = 0
                        for j in range(cols):
                            if j == k:
                                continue
                            
                            coeff = 1 - (alpha[j] // alpha[k])
                            
                            for i in range(n * m):
                                new_U[i][nc] = U[i][j] * coeff
                            for i in range(m * p):
                                new_V[i][nc] = V[i][j]
                            for i in range(p * n):
                                new_W[i][nc] = W[i][j]
                            
                            nc += 1
                        
                        cand = {"U": new_U, "V": new_V, "W": new_W}
                        
                        if _validate(cand):
                            best_candidate = cand
                            best_rank = new_rank
                            eliminated = True
                            break
                if eliminated:
                    break
            
            if not eliminated:
                break

    if _elapsed_ok():
        _try_reduce_loop()

    return best_candidate

# EVOLVE-BLOCK-END
