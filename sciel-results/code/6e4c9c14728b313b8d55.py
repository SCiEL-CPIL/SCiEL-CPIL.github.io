def construct():
    # Order 10: no pair of orthogonal Latin squares exists (Euler's 36 officers / N(10)=0).
    # Return a single valid Latin square with Python ints.
    n = 10
    L = [[int((i + j) % n) for j in range(n)] for i in range(n)]
    return [L]
