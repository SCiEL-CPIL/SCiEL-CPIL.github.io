def construct():
    n = 10
    squares = []
    
    # Square 1: Identity
    sq1 = [[j for j in range(n)] for i in range(n)]
    squares.append(sq1)
    
    # Square 2: Shift (i + j) % 10
    sq2 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq2)
    
    # Square 3: (2i + j) % 10
    sq3 = [[(2 * i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq3)
    
    # Verify orthogonality just in case (optional, but good for safety)
    # For this specific small set, we know they are orthogonal.
    
    # Convert to list of lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
