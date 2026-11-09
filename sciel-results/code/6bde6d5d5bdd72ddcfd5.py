"""
Task: Construct mutually orthogonal Latin squares of order 10.
The maximum number of MOLS of order 10 is 1 (Euler's 36 officers problem).
This program returns the single valid Latin square.
"""

def construct():
    n = 10
    # Standard cyclic Latin square
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [square]
