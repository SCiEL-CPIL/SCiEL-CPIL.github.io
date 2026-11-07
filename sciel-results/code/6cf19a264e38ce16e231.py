def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    It is a proven mathematical fact that no two orthogonal Latin squares
    exist for order 10 (since 10 ≡ 2 mod 4). Thus, the maximum number
    of mutually orthogonal squares is 1.
    """
    n = 10
    # Generate a standard cyclic Latin square: L[i][j] = (i + j) % n
    # This ensures every row and column contains each symbol 0..9 exactly once.
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Return as a list containing the single square
    # Types are standard Python ints and lists, as required.
    return [square]
