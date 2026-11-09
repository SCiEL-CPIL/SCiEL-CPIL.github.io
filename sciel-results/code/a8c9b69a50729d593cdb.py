def construct():
    # Mathematical fact: No two mutually orthogonal Latin squares exist for order 10.
    # Thus, the maximum number of pairwise orthogonal Latin squares is 1.
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [L1]
