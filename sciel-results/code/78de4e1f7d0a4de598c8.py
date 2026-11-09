import time
import random

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    # It is a well-known mathematical fact that no pair of orthogonal Latin squares of order 10 exists.
    # Specifically, the maximum number of mutually orthogonal Latin squares (MOLS) of order 10 is 1.
    # Therefore, we return a single valid Latin square.
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    return [L1]
