"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_half_int(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _solve_linear_system(A, b):
    """Solve A*x = b using Gaussian elimination with Fraction. Returns None if no solution."""
    n = len(A)
    if n == 0:
        return []
    cols = len(A[0])
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    rank = 0
    for col in range(cols):
        if rank >= n:
            break
        # Find pivot
        pivot = None
        for row in range(rank, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        M[rank], M[pivot] = M[pivot], M[rank]
        # Eliminate
        for row in range(n):
            if row != rank and M[row][col] != 0:
                factor = M[row][col] / M[rank][col]
                for c in range(col, cols + 1):
                    M[row][c] -= factor * M[rank][c]
        rank += 1
    # Check for inconsistency
    for row in range(rank, n):
        if M[row][cols] != 0:
            return None
    # Extract solution (free variables = 0)
    x = [Fraction(0)] * cols
    for i in range(rank):
        x[i] = M[i][cols]
    return x

def _try_rank_slice(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition using the slice method.
    Fix U and V (sparse patterns), solve for W.
    """
    # For each column c, U has entries at some (i,j) positions, V at some (j,k) positions.
    # W[k*n+i] must satisfy: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: try sparse U and V with 0/1 entries
    # For each column c, pick a subset of (i,j) for U and subset of (j,k) for V
    # Then W is determined by linear system
    
    # Simplified: try rank where each column contributes to specific patterns
    # For small ranks, enumerate sparse structures
    
    # For rank r, we need r columns. Each column c has:
    #   U_c: support on some (i,j) pairs
    #   V_c: support on some (j,k) pairs  
    #   W_c: support on some (k,i) pairs
    # Constraint: for each (i,j,k), exactly the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # A natural sparse structure: each column c corresponds to a "path" i->j->k
    # where U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i)
    # But we can have more general structures.
    
    # Let's try: for each column, U has exactly one 1, V has exactly one 1, W has exactly one 1
    # This is the schoolbook pattern. To do better, allow multiple 1s per column.
    
    # For rank r < n*m*p, we need some columns to cover multiple (i,j,k) triples.
    
    # Approach: fix U to have specific patterns, fix V, solve for W
    # Try U where each row has at most 1 nonzero, V where each row has at most 1 nonzero
    
    # Let's try a more systematic approach:
    # Fix U as n*m x r with 0/1 entries, each row sum <= 1
    # Fix V as m*p x r with 0/1 entries, each row sum <= 1
    # Solve for W as p*n x r
    
    # For each (k,i), the W row is W[k*n+i, :]
    # The constraint for (i,j,k) is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W[k*n+i, :] for fixed i,j,k
    
    # Group by (k,i): for fixed (k,i), the constraints over all j give:
    # sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1  for each j... 
    # Wait, that's not right. For fixed (k,i), we have constraints for each j:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of m equations in r unknowns (W[k*n+i, 0..r-1])
    
    # So for each (k,i), we have a linear system A_{k,i} * W_{k,i} = 1 (vector of m ones)
    # where A_{k,i}[j, c] = U[i*m+j, c] * V[j*p+k, c]
    
    # Let's try small r values
    
    # First, try r = n*m*p (schoolbook) as baseline
    # Then try lower ranks
    
    # For efficiency, let's try specific structured U and V
    
    # Pattern 1: U[i*m+j, c] = 1 if c == i*m+j (identity-like), V[j*p+k, c] = 1 if c == j*p+k
    # Then A_{k,i}[j, c] = 1 if c == i*m+j AND c == j*p+k, which requires i*m+j == j*p+k
    # This only works when the indices align
    
    # Let me try a different approach: use random sparse U, V and solve for W
    
    import random
    random.seed(42)
    
    best_result = None
    
    # Try ranks from a lower bound up
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually rank >= max(n, m, p) * min(n, m, p) for matrix mult? No.
    # Known lower bound for 3x3 x 3x5: at least 15 (since output is 3x5=15)
    # Let's try from 15 upward
    
    for r in range(15, 46):
        if time.time() > deadline:
            break
        
        # Try several random sparse patterns for U and V
        attempts = 50 if r <= 20 else 20
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse U: n*m x r
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            
            # Each row of U gets 1-2 nonzeros, each row of V gets 1-2 nonzeros
            for row in range(n * m):
                num_nonzero = random.randint(1, min(2, r))
                cols = random.sample(range(r), num_nonzero)
                for c in cols:
                    U[row][c] = 1
            
            for row in range(m * p):
                num_nonzero = random.randint(1, min(2, r))
                cols = random.sample(range(r), num_nonzero)
                for c in cols:
                    V[row][c] = 1
            
            # Solve for W
            W = [[0] * r for _ in range(p * n)]
            solvable = True
            
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    A = []
                    b = []
                    for j in range(m):
                        row_A = []
                        for c in range(r):
                            row_A.append(Fraction(U[i * m + j][c] * V[j * p + k][c]))
                        A.append(row_A)
                        b.append(Fraction(1))
                    
                    # Solve A * w = b where w = W[k*n+i, :]
                    x = _solve_linear_system(A, b)
                    if x is None:
                        solvable = False
                        break
                    W[k * n + i] = [int(val) if val.denominator == 1 else float(val) for val in x]
                
                if not solvable:
                    break
            
            if solvable:
                # Verify
                if _verify_half_int(U, V, W):
                    best_result = {"U": U, "V": V, "W": W, "rank": r}
                    return best_result
    
    return best_result

def construct():
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # Try to find a low-rank decomposition
    result = _try_rank_slice(3, 3, 5, 0, deadline)
    
    if result is not None and time.time() < deadline:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
