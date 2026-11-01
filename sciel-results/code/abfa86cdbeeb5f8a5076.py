def solve_linear_system_exact(A, b):
    """Solve A*x = b where A is m x n matrix of Fractions, b is list of m Fractions.
    Returns solution x as list of n Fractions, or None if no solution."""
    m = len(A)
    n = len(A[0])
    
    # Augmented matrix
    aug = [[A[i][j] for j in range(n)] + [b[i]] for i in range(m)]
    
    # Gaussian elimination
    row = 0
    for col in range(n):
        if row >= m:
            break
        # Find pivot
        pivot = None
        for r in range(row, m):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        aug[row], aug[pivot] = aug[pivot], aug[row]
        # Normalize
        pv = aug[row][col]
        for j in range(col, n+1):
            aug[row][j] = aug[row][j] / pv
        # Eliminate
        for r in range(m):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for j in range(col, n+1):
                    aug[r][j] = aug[r][j] - factor * aug[row][j]
        row += 1
    
    # Check for inconsistency
    for r in range(row, m):
        if aug[r][n] != 0:
            return None
    
    # Free variables set to 0
    x = [Fraction(0)] * n
    for r in range(row):
        # Find pivot column for this row
        for col in range(n):
            if aug[r][col] != 0:
                x[col] = aug[r][n]
                break
    
    return x
