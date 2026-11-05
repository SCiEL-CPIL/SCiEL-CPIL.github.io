import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # Generate Paley(29) coloring
    n = 29
    q = n
    squares = {x*x % q for x in range(1, q)}
    
    # Precompute colors in lexicographic order
    # Edges (i, j) for i < j
    # Total edges: 29*28/2 = 406
    colors = np.zeros(n*(n-1)//2, dtype=np.int8)
    idx = 0
    for i in range(q):
        for j in range(i+1, q):
            d = (j - i) % q
            if d in squares:
                colors[idx] = 0
            else:
                colors[idx] = 1
            idx += 1
            
    # We can attempt to extend to 30 if we have time, but it's risky.
    # Given the difficulty, returning 29 is the robust solution.
    
    return {'n': n, 'colors': colors.tolist()}
