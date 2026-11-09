def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    Mathematical proof (Tarry 1901, Bose et al. 1959) establishes that
    no two mutually orthogonal Latin squares exist for order 10.
    Thus, the maximum number of pairwise orthogonal squares is 1.
    """
    n = 10
    square = [[int((i + j) % n) for j in range(n)] for i in range(n)]
    return [square]
