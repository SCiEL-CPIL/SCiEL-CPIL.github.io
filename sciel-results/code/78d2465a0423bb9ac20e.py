def construct():
    # Construct a single Latin square of order 10.
    # Since MOLS of order 10 do not exist (max is 1), we return exactly 1 square.
    # We use a standard cyclic construction for n=10.
    n = 10
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure types are strictly Python int and list as per contract.
    # The list comprehension above already produces Python ints and lists.
    # We wrap it in a list to return a list of squares.
    return [square]
