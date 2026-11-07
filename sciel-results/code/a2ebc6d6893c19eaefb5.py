def construct():
    n = 10
    # Construct a valid Latin square of order n
    # Using the standard cyclic group construction
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # The problem asks to maximize the number of pairwise orthogonal arrays.
    # For n=10, it is known that no two Latin squares are orthogonal.
    # Thus, the maximum number is 1.
    # We return a list containing this single square.
    
    # Ensure types are Python int
    # The list comprehension above produces Python ints.
    
    return [square]
