"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    # Construct the standard cyclic Latin square of order 10.
    # For n=10, it is impossible to have any pair of orthogonal Latin squares.
    # Thus, the maximum size of a set of MOLS(10) is 1.
    square = [[int((i + j) % n) for j in range(n)] for i in range(n)]
    
    # Return a list containing this single square.
    # Ensure all elements are Python integers and lists are standard lists.
    return [square]

# EVOLVE-BLOCK-END
