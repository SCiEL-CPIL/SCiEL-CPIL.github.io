def construct():
    S = set()
    # 4x8 grid structure
    # i in 1..4, j in 1..8
    # value = i + 10*j
    for i in range(1, 5):
        for j in range(1, 9):
            S.add(i + 10 * j)
    return sorted(S)
