import time

def construct():
    n = 10
    # L1(i,j) = (i + j) % 10 is a valid Latin square.
    # It is known that the maximum number of MOLS of order 10 is 1.
    # Therefore, we return a single valid Latin square.
    
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure types are Python int
    # The list comprehension produces Python ints.
    
    return [square]
