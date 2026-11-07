"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Square 1: Standard cyclic
    square1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2: Constructed using CRT components
    # L2(i,j) = 5*(i%2) + 6*((i%5 + 2*(j%5))%5) % 10
    square2 = []
    for i in range(n):
        row = []
        for j in range(n):
            i2 = i % 2
            i5 = i % 5
            j5 = j % 5
            v5 = (i5 + 2 * j5) % 5
            val = (5 * i2 + 6 * v5) % 10
            row.append(val)
        square2.append(row)
        
    return [square1, square2]
# EVOLVE-BLOCK-END
