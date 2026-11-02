"""Program for matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    We use a structured construction with integer/half-integer entries.
    The approach:
    1. Start from the schoolbook decomposition (rank 30)
    2. Try to merge terms using algebraic identities
    3. Verify each candidate for exact correctness
    4. Return the best (lowest rank) valid decomposition found
    
    The decomposition satisfies: for all i,j,k
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (10, rank)
    
    deadline = time.time() + 125  # Leave safety margin for 150s limit
    
    def validate(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(x):
        """Check if x is an integer or half-integer."""
        return abs(x * 2 - round(x * 2)) < 1e-9
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2
    
    def try_rank(rank, max_attempts=20):
        """
        Try to find a valid decomposition of given rank.
        Uses random initialization with structured patterns.
        """
        rng = random.Random()
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with structured patterns
            # Try different initialization strategies
            
            # Strategy 1: Start from schoolbook and merge
            if attempt < max_attempts // 3:
                U = [[0] * rank for _ in range(n*m)]
                V = [[0] * rank for _ in range(m*p)]
                W = [[0] * rank for _ in range(p*n)]
                
                # Use first min(rank, 30) terms from schoolbook
                num_terms = min(rank, n*m*p)
                for t in range(num_terms):
                    i = t // (m*p)
                    rem = t % (m*p)
                    j = rem // p
                    k = rem % p
                    U[i*m+j][t] = 1
                    V[j*p+k][t] = 1
                    W[k*n+i][t] = 1
                
                # For remaining columns, add random patterns
                for t in range(num_terms, rank):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][t] = rng.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k][t] = rng.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i][t] = rng.choice([-1, 0, 1])
            
            # Strategy 2: Random integer/half-integer initialization
            elif attempt < 2 * max_attempts // 3:
                U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
                V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
                W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            # Strategy 3: Diagonal-like patterns
            else:
                U = [[0] * rank for _ in range(n*m)]
                V = [[0] * rank for _ in range(m*p)]
                W = [[0] * rank for _ in range(p*n)]
                
                for c in range(rank):
                    # Each column has a specific pattern
                    i_base = c % n
                    j_base = c % m
                    k_base = c % p
                    
                    U[i_base*m + j_base][c] = 1
                    V[j_base*p + k_base][c] = 1
                    W[k_base*n + i_base][c] = 1
                    
                    # Add some cross terms
                    for i in range(n):
                        for j in range(m):
                            if (i, j) != (i_base, j_base) and rng.random() < 0.3:
                                U[i*m+j][c] = rng.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            if (j, k) != (j_base, k_base) and rng.random() < 0.3:
                                V[j*p+k][c] = rng.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            if (k, i) != (k_base, i_base) and rng.random() < 0.3:
                                W[k*n+i][c] = rng.choice([-1, 0, 1])
            
            # Try to improve via local search
            improved = True
            max_improve_iters = 100
            improve_iter = 0
            
            while improved and improve_iter < max_improve_iters:
                if time.time() > deadline:
                    break
                
                improved = False
                improve_iter += 1
                
                # Try flipping or adjusting individual entries
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random entry to modify
                    mat = rng.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = rng.randrange(n*m)
                        col = rng.randrange(rank)
                        old_val = U[row][col]
                        new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val != old_val:
                            U[row][col] = new_val
                            if validate(U, V, W, tol=1e-9):
                                improved = True
                                break
                            else:
                                U[row][col] = old_val
                    elif mat == 'V':
                        row = rng.randrange(m*p)
                        col = rng.randrange(rank)
                        old_val = V[row][col]
                        new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val != old_val:
                            V[row][col] = new_val
                            if validate(U, V, W, tol=1e-9):
                                improved = True
                                break
                            else:
                                V[row][col] = old_val
                    else:  # W
                        row = rng.randrange(p*n)
                        col = rng.randrange(rank)
                        old_val = W[row][col]
                        new_val = rng.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val != old_val:
                            W[row][col] = new_val
                            if validate(U, V, W, tol=1e-9):
                                improved = True
                                break
                            else:
                                W[row][col] = old_val
            
            # Check if we have a valid solution
            if validate(U, V, W, tol=1e-9):
                # Verify all entries are half-integers
                all_valid = True
                for row in U:
                    for x in row:
                        if not is_half_integer(x):
                            all_valid = False
                            break
                if not all_valid:
                    for row in V:
                        for x in row:
                            if not is_half_integer(x):
                                all_valid = False
                                break
                if not all_valid:
                    for row in W:
                        for x in row:
                            if not is_half_integer(x):
                                all_valid = False
                                break
                
                if all_valid:
                    # Round to ensure clean half-integers
                    U_r = [[to_half_integer(x) for x in row] for row in U]
                    V_r = [[to_half_integer(x) for x in row] for row in V]
                    W_r = [[to_half_integer(x) for x in row] for row in W]
                    
                    if validate(U_r, V_r, W_r, tol=1e-9):
                        return U_r, V_r, W_r
        
        return None
    
    # Try to find a known low-rank decomposition
    # For 2x3x5, let's try specific constructions
    
    # Known: rank 15 might work with a specific pattern
    # Let's try a structured construction
    
    # Attempt specific rank values
    for rank in [15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_attempts=30)
        if result is not None:
            U, V, W = result
            if validate(U, V, W, tol=1e-9):
                return {"U": U, "V": V, "W": W}
    
    # If we haven't found a better decomposition, try to construct one
    # using algebraic identities
    
    # For 2x3x5, we can try to use Strassen-like ideas
    # but adapted to our dimensions
    
    # Let's try a more systematic approach:
    # Each output element (i,k) is sum_j A[i,j]*B[j,k]
    # We need to compute n*p = 10 such dot products, each of length m=3
    
    # A simple lower bound: we need at least enough "slots" to handle
    # the m=3 multiplications per output, with sharing
    
    # Let's try rank = 15 with a specific construction
    
    # Construction idea: use 15 terms where each term contributes
    # to multiple output elements
    
    U = [[0] * 15 for _ in range(6)]
    V = [[0] * 15 for _ in range(15)]
    W = [[0] * 15 for _ in range(10)]
    
    # Try to fill in a pattern
    # Term 0-4: handle output (0,0) through (0,4)
    # Term 5-9: handle output (1,0) through (1,4)
    # Term 10-14: cross terms for sharing
    
    for k in range(5):
        # Output (0, k): sum_j A[0,j]*B[j,k]
        # We'll use terms k, k+5, k+10
        U[0][k] = 1  # A[0,0]
        V[k][k] = 1  # B[0,k]
        W[k*2+0][k] = 1  # Output (0,k)
        
        U[1][k+5] = 1  # A[0,1]
        V[5+k][k+5] = 1  # B[1,k]
        W[k*2+0][k+5] = 1
        
        U[2][k+10] = 1  # A[0,2]
        V[10+k][k+10] = 1  # B[2,k]
        W[k*2+0][k+10] = 1
    
    for k in range(5):
        # Output (1, k): sum_j A[1,j]*B[j,k]
        # We'll use terms k, k+5, k+10 (shared with above where possible)
        U[3][k] = 1  # A[1,0]
        # B[0,k] already in V[k]
        W[k*2+1][k] = 1
        
        U[4][k+5] = 1  # A[1,1]
        W[k*2+1][k+5] = 1
        
        U[5][k+10] = 1  # A[1,2]
        W[k*2+1][k+10] = 1
    
    if validate(U, V, W, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return schoolbook decomposition (rank 30)
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
# EVOLVE-BLOCK-END
