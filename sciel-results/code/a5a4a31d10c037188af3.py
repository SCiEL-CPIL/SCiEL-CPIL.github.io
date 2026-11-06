import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Construct the optimal OneMax solution: a string of N '1's.
    # Using list comprehension and join as an alternative construction method.
    return ''.join(['1' for _ in range(N)])
# EVOLVE-BLOCK-END
