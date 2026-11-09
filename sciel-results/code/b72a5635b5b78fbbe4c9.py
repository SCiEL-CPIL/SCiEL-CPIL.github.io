import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # The smallest known valid tournament for the 4-dominator property is the
    # Paley tournament on 67 vertices. We construct it directly.
    p = 67

    # Compute quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)

    # Build orientation list: for each pair (i, j) with i < j,
    # 0 means i -> j (i dominates j), 1 means j -> i
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)

    return {'n': p, 'orientation': orientation}
