<<<<<<< SEARCH
    def try_rank_r(r, num_attempts=20):
        """Try to find a rank-r decomposition using structured search."""
        # Strategy: Start with schoolbook and try to merge terms
        # For rank 44, we need to save 1 term from the 45 schoolbook terms.
        # 
        # Idea: Use a "Strassen-like" approach where we introduce a shared variable.
        # 
        # Let's try a different approach: parameterize U and V with small integer/rational
        # entries and solve for W exactly.
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly choose a pattern: most entries are 0 or 1, some are -1 or 2
            # to allow for Strassen-like cancellations
            U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
            
            # Initialize with schoolbook-like pattern but with some modifications
            # For each column c, assign it to a (i,j,k) triple, but allow sharing
            col_assignments = []
            for c in range(r):
                # Randomly assign to a (i,j,k) with some probability of sharing
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                col_assignments.append((i, j, k))
            
            # Set U and V based on assignments
            for c, (i, j, k) in enumerate(col_assignments):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
            
            # Now solve for W exactly
            # For each (i,k), we have m equations:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for j=0..m-1
            # This is a linear system in W[k*n+i, 0..r-1]
            
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = [[Fraction(0) for _ in range(r)] for _ in range(m)]
                    for j in range(m):
                        for c in range(r):
                            A[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # Solve A * w = 1_m where w = W[k*n+i, :]
                    # Use exact Gaussian elimination
                    # Augmented matrix: A | 1
                    aug = [row[:] + [Fraction(1)] for row in A]
                    
                    # Gaussian elimination
                    rows = m
                    cols = r
                    pivot_row = 0
                    rank = 0
                    for col in range(cols):
                        # Find pivot
                        pivot = -1
                        for row in range(pivot_row, rows):
                            if aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        # Swap
                        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
                        # Eliminate below
                        for row in range(pivot_row + 1, rows):
                            if aug[row][col] != 0:
                                factor = aug[row][col] / aug[pivot_row][col]
                                for cc in range(col, cols + 1):
                                    aug[row][cc] -= factor * aug[pivot_row][cc]
                        pivot_row += 1
                        rank += 1
                    
                    # Check consistency: any row that is all zeros in A part must have 0 in RHS
                    consistent = True
                    for row in range(pivot_row, rows):
                        all_zero = True
                        for col in range(cols):
                            if aug[row][col] != 0:
                                all_zero = False
                                break
                        if all_zero and aug[row][cols] != 0:
                            consistent = False
                            break
                    
                    if not consistent:
                        valid = False
                        break
                    
                    # Back substitution to find one solution
                    w_sol = [Fraction(0) for _ in range(cols)]
                    for row in range(pivot_row - 1, -1, -1):
                        val = aug[row][cols]
                        for cc in range(row + 1, cols):
                            val -= aug[row][cc] * w_sol[cc]
                        if aug[row][row] != 0:
                            w_sol[row] = val / aug[row][row]
                        else:
                            # Free variable, set to 0
                            w_sol[row] = Fraction(0)
                    
                    W[k * n + i] = w_sol
            
            if valid:
                if verify(U, V, W, r):
                    return U, V, W
        
        return None
=======
    def try_rank_r(r, num_attempts=50):
        """Try to find a rank-r decomposition using structured search."""
        def solve_for_W(U, V, r):
            """Given U and V, solve for W exactly. Returns W or None if inconsistent."""
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            for i in range(n):
                for k in range(p):
                    # Build the m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = [[Fraction(0) for _ in range(r)] for _ in range(m)]
                    for j in range(m):
                        for c in range(r):
                            A[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # Solve A * w = 1_m where w = W[k*n+i, :]
                    aug = [row[:] + [Fraction(1)] for row in A]
                    rows = m
                    cols = r
                    pivot_row = 0
                    for col in range(cols):
                        pivot = -1
                        for row in range(pivot_row, rows):
                            if aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
                        for row in range(pivot_row + 1, rows):
                            if aug[row][col] != 0:
                                factor = aug[row][col] / aug[pivot_row][col]
                                for cc in range(col, cols + 1):
                                    aug[row][cc] -= factor * aug[pivot_row][cc]
                        pivot_row += 1
                    
                    for row in range(pivot_row, rows):
                        all_zero = all(aug[row][col] == 0 for col in range(cols))
                        if all_zero and aug[row][cols] != 0:
                            return None
                    
                    w_sol = [Fraction(0) for _ in range(cols)]
                    for row in range(pivot_row - 1, -1, -1):
                        val = aug[row][cols]
                        for cc in range(row + 1, cols):
                            val -= aug[row][cc] * w_sol[cc]
                        if aug[row][row] != 0:
                            w_sol[row] = val / aug[row][row]
                    W[k * n + i] = w_sol
            return W

        def generate_schoolbook_with_merges(r):
            """Generate U, V by starting from schoolbook and merging (r-45) terms."""
            # Start with all 45 schoolbook columns
            U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
            
            # First 45 columns: schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = Fraction(1)
                        V[j * p + k][col] = Fraction(1)
                        col += 1
            
            # For remaining columns (if r > 45), add random entries
            for c in range(45, r):
                # Random sparse pattern
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    U[i * m + j][c] = Fraction(random.choice([-1, 1, 2]))
                for _ in range(random.randint(1, 3)):
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    V[j * p + k][c] = Fraction(random.choice([-1, 1, 2]))
            
            return U, V

        def generate_strassen_like(r):
            """Generate U, V with Strassen-like structure for reduced rank."""
            U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
            
            # Use a structured pattern: group columns by which (i,j) or (j,k) they affect
            # For rank r, we have 45 - r "savings" to achieve
            savings = 45 - r
            
            # Assign each column to a primary (i,j,k) but allow some to be shared
            assignments = []
            for c in range(r):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                assignments.append((i, j, k))
            
            # Set base values
            for c, (i, j, k) in enumerate(assignments):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
            
            # Add some non-trivial entries to enable cancellations
            num_extra = min(savings * 2, r // 2)
            for _ in range(num_extra):
                c = random.randint(0, r - 1)
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                if U[i * m + j][c] == 0:
                    U[i * m + j][c] = Fraction(random.choice([-1, 1]))
                j2 = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                if V[j2 * p + k][c] == 0:
                    V[j2 * p + k][c] = Fraction(random.choice([-1, 1]))
            
            return U, V

        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Choose strategy based on attempt number
            if attempt % 3 == 0:
                U, V = generate_schoolbook_with_merges(r)
            elif attempt % 3 == 1:
                U, V = generate_strassen_like(r)
            else:
                # Pure random with small entries
                U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
                V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
                for c in range(r):
                    # Each column affects 1-2 entries in U
                    for _ in range(random.randint(1, 2)):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        U[i * m + j][c] = Fraction(random.choice([-1, 0, 1]))
                    # Each column affects 1-2 entries in V
                    for _ in range(random.randint(1, 2)):
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        V[j * p + k][c] = Fraction(random.choice([-1, 0, 1]))
            
            W = solve_for_W(U, V, r)
            if W is not None:
                if verify(U, V, W, r):
                    return U, V, W
        
        return None
>>>>>>> REPLACE

