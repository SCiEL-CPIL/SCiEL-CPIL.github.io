import random

def construct():
    n = 10
    # Base Latin Square: L[i][j] = (i + j) % n
    # This is a valid Latin Square.
    
    # To make it a "new approach" (Algorithmic/Shuffled), we will
    # generate a random isomorphic copy of this square.
    # This ensures the output is not trivially the identity-like square,
    # though it is mathematically equivalent.
    
    # 1. Random row permutation
    row_perm = random.sample(range(n), n)
    # 2. Random column permutation
    col_perm = random.sample(range(n), n)
    # 3. Random symbol permutation
    sym_perm = random.sample(range(n), n)
    
    # Construct the new square
    L = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Map logical row i to physical row row_perm[i]
            # Map logical col j to physical col col_perm[j]
            # The base value at (row_perm[i], col_perm[j]) is (row_perm[i] + col_perm[j]) % n
            val = (row_perm[i] + col_perm[j]) % n
            # Relabel symbol
            L[i][j] = sym_perm[val]
            
    # Verification (optional but good for safety in "new" code)
    # Check rows
    for i in range(n):
        if sorted(L[i]) != list(range(n)):
            # Fallback to standard if something went wrong (shouldn't happen)
            return [[[ (i + j) % n for j in range(n)] for i in range(n)]]
    # Check cols
    for j in range(n):
        col = [L[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return [[[ (i + j) % n for j in range(n)] for i in range(n)]]
            
    # Ensure Python ints
    result = [[int(val) for val in row] for row in L]
    
    return [result]
