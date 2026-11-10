def construct():
    n = 10
    squares = []
    for k in range(1, 10):
        sq = []
        for i in range(n):
            row = []
            for j in range(n):
                v = (i + k * j) % 11
                if v == 0:
                    row.append(9)
                else:
                    row.append(v - 1)
            sq.append(row)
        squares.append(sq)
    return squares
