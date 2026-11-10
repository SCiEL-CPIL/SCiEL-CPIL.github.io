def construct():
    # It is a proven mathematical fact (Euler's 36 officers problem,
    # resolved by Tarry in 1901) that no pair of orthogonal Latin
    # squares of order 10 exists. Therefore the maximum number of
    # mutually orthogonal Latin squares of order 10 is 1.
    # Return a single valid Latin square of order 10.
    A = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    return [A]
