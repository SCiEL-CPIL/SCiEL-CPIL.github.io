return [list(p) for p in [(i, 0) for i in range(n)] if dimension == 2 else [(i,) + (0,) * (dimension - 1) for i in range(n)]]
