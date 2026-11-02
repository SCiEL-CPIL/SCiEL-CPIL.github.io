import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    def is_nice_frac(val, tol=1e-6):
        """Check if val is close to a half-integer (k/2 for integer k)"""
        x2 = val * 2
        r = round(x2)
        return abs(x2 - r) < tol

    def to_half_int(val, tol=1e-6):
        """Convert to nearest half-integer, return None if not close"""
        x2 = val * 2
        r = round(x2)
        if abs(x2 - r) < tol:
            return r / 2.0
        return None

    def solve_for_w(U, V, r, tol=1e-8):
        """
        Given U (nm x r) and V (mp x r), solve for W (pn x r) such that
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        Returns W as list of lists of half-integers, or None if no nice solution.
        """
        W = [[0.0] * r for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                # Build system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j][c] * V[j * p + k][c]
                
                try:
                    # Solve A @ w = ones(m)
                    w_sol, residuals, rank, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check if solution is close to half-integers
                    w_rounded = []
                    for c in range(r):
                        val = to_half_int(w_sol[c], tol=1e-6)
                        if val is None:
                            return None
                        w_rounded.append(val)
                    
                    for c in range(r):
                        W[k * n + i][c] = w_rounded[c]
                except Exception:
                    return None
        
        return W

    def verify_decomposition(U, V, W, r, tol=1e-6):
        """Verify that the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def rank1_perturb_search(r, trials, seed_offset=0):
        """
        Search for rank-r decompositions by starting from schoolbook and applying
        rank-1 perturbations. This is based on the idea that low-rank matrix multiplication
        schemes can often be found by modifying the schoolbook algorithm.
        """
        for trial in range(trials):
            if time.time() > deadline:
                return None
            
            random.seed(42 + seed_offset + trial)
            
            # Start with schoolbook-like structure but with r columns
            # We'll create U and V with a mix of structured and random entries
            
            # Strategy: Use a base pattern with some randomness
            # For each column c, decide which (i,j), (j,k) pairs are active
            
            U = [[0.0] * r for _ in range(n * m)]
            V = [[0.0] * r for _ in range(m * p)]
            
            for c in range(r):
                # Randomly choose which entries are nonzero in column c
                # Use a structured approach: pick random (i,j) and (j',k) pairs
                
                # Method 1: Pure random sparse
                if random.random() < 0.3:
                    # Random sparse with density ~0.3
                    for i in range(n * m):
                        if random.random() < 0.3:
                            U[i][c] = random.choice([-1, 1])
                    for i in range(m * p):
                        if random.random() < 0.3:
                            V[i][c] = random.choice([-1, 1])
                
                # Method 2: Structured - pick one (i,j) and one (j,k)
                elif random.random() < 0.6:
                    i_idx = random.randint(0, n * m - 1)
                    j_idx = random.randint(0, m * p - 1)
                    U[i_idx][c] = random.choice([-1, 1])
                    V[j_idx][c] = random.choice([-1, 1])
                
                # Method 3: Multiple structured entries
                else:
                    num_u = random.randint(1, 3)
                    num_v = random.randint(1, 3)
                    for _ in range(num_u):
                        i_idx = random.randint(0, n * m - 1)
                        U[i_idx][c] = random.choice([-1, 1])
                    for _ in range(num_v):
                        j_idx = random.randint(0, m * p - 1)
                        V[j_idx][c] = random.choice([-1, 1])
            
            W = solve_for_w(U, V, r)
            if W is not None:
                if verify_decomposition(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
        
        return None

    def structured_search(r, trials, seed_offset=0):
        """
        More structured search: use patterns based on known low-rank techniques.
        For small matrices, we can try specific patterns.
        """
        for trial in range(trials):
            if time.time() > deadline:
                return None
            
            random.seed(1000 + seed_offset + trial)
            
            U = [[0.0] * r for _ in range(n * m)]
            V = [[0.0] * r for _ in range(m * p)]
            
            # Try to create columns that cover the necessary products
            # Each column c should help compute some subset of the output entries
            
            for c in range(r):
                # Choose a random "pattern" for this column
                pattern_type = random.choice(['single', 'pair', 'cross', 'random'])
                
                if pattern_type == 'single':
                    # One U entry and one V entry
                    ij = random.randint(0, n * m - 1)
                    jk = random.randint(0, m * p - 1)
                    U[ij][c] = random.choice([-1, 1])
                    V[jk][c] = random.choice([-1, 1])
                
                elif pattern_type == 'pair':
                    # Two U entries sharing same j, two V entries
                    j = random.randint(0, m - 1)
                    i1 = random.randint(0, n - 1)
                    i2 = random.randint(0, n - 1)
                    k1 = random.randint(0, p - 1)
                    k2 = random.randint(0, p - 1)
                    U[i1 * m + j][c] = random.choice([-1, 1])
                    U[i2 * m + j][c] = random.choice([-1, 1])
                    V[j * p + k1][c] = random.choice([-1, 1])
                    V[j * p + k2][c] = random.choice([-1, 1])
                
                elif pattern_type == 'cross':
                    # Cross pattern
                    i1 = random.randint(0, n - 1)
                    j1 = random.randint(0, m - 1)
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    k1 = random.randint(0, p - 1)
                    k2 = random.randint(0, p - 1)
                    U[i1 * m + j1][c] = random.choice([-1, 1])
                    U[i2 * m + j2][c] = random.choice([-1, 1])
                    V[j1 * p + k1][c] = random.choice([-1, 1])
                    V[j2 * p + k2][c] = random.choice([-1, 1])
                
                else:
                    # Random
                    for i in range(n * m):
                        if random.random() < 0.2:
                            U[i][c] = random.choice([-1, 1])
                    for i in range(m * p):
                        if random.random() < 0.2:
                            V[i][c] = random.choice([-1, 1])
            
            W = solve_for_w(U, V, r)
            if W is not None:
                if verify_decomposition(U, V, W, r):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Search strategy: try ranks from 59 down to 40
    # Use multiple search methods
    for r in range(59, 39, -1):
        if time.time() > deadline:
            break
        
        # Allocate time budget for this rank
        time_per_rank = (deadline - time.time()) / max(1, (59 - r + 1))
        
        # Method 1: Random sparse search
        trials1 = min(300, int(time_per_rank / 0.05))
        result = rank1_perturb_search(r, trials1, seed_offset=r * 100)
        if result is not None:
            best_result = result
            best_rank = r
            continue  # Try lower rank
        
        # Method 2: Structured search
        if time.time() < deadline:
            trials2 = min(200, int((deadline - time.time()) / 0.05))
            result = structured_search(r, trials2, seed_offset=r * 200)
            if result is not None:
                best_result = result
                best_rank = r
                continue  # Try lower rank
    
    return best_result
# EVOLVE-BLOCK-END
