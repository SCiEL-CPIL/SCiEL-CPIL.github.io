def construct():
    """
    Construct a 12-state binary automaton that synchronizes, maximizing the reset threshold.
    Returns the known optimal Černý automaton for N=12, which has a reset threshold of 121.
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[0] = 1
    for i in range(1, N):
        b[i] = i - 1
    return [a, b]
