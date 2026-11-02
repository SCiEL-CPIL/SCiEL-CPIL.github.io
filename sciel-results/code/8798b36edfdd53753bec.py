q = k # 3
d = g - 5 # 11
# rule definition
rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
for c in range(5, d + 1):
    r = (c - 5) % 4
    c0 = c - r
    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

# BFS
start = (0, (None,) + (0,) * d)
# ...
for first in range(q):
    y = [None, first] + [0] * (d - 1)
    for c in range(2, d + 1):
        a, b = rule[c]
        if side == 0:
            y[c] = add(x[c], mul(y[a], x[b]))
        else:
            y[c] = sub(x[c], mul(x[a], y[b]))
    neighbor = (1 - side, tuple(y))
