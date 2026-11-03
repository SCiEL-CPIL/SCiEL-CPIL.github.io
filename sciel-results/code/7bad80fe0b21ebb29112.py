import math
import itertools
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Try to find a circulant graph C_N(a, b) with girth >= 10
    # Use NumPy for faster girth check
    
    def check_girth_fast(N, a, b):
        gens = np.array([a, -a, b, -b]) % N
        inv_idx = np.array([1, 0, 3, 2])
        
        # Length 4
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        if (gens[i1] + gens[i2] + gens[i3] + gens[i4]) % N == 0:
                            return False
                            
        # Length 6
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        for i5 in range(4):
                            if i5 == inv_idx[i4]: continue
                            for i6 in range(4):
                                if i6 == inv_idx[i5]: continue
                                if (gens[i1] + gens[i2] + gens[i3] + gens[i4] + gens[i5] + gens[i6]) % N == 0:
                                    return False
                                    
        # Length 8
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        s4 = gens[i1] + gens[i2] + gens[i3] + gens[i4]
                        for i5 in range(4):
                            if i5 == inv_idx[i4]: continue
                            s5 = s4 + gens[i5]
                            for i6 in range(4):
                                if i6 == inv_idx[i5]: continue
                                s6 = s5 + gens[i6]
                                for i7 in range(4):
                                    if i7 == inv_idx[i6]: continue
                                    s7 = s6 + gens[i7]
                                    for i8 in range(4):
                                        if i8 == inv_idx[i7]: continue
                                        if (s7 + gens[i8]) % N == 0:
                                            return False
        return True

    # Search
    for N in range(243, 500):
        if N % 2 != 0:
            pass
        
        found = False
        for a in range(1, N // 2 + 1):
            if 2 * a % N == 0: continue
            for b in range(a + 1, N // 2 + 1):
                if 2 * b % N == 0: continue
                if (a + b) % N == 0: continue
                if (a - b) % N == 0: continue
                if (b - a) % N == 0: continue
                if a == b: continue
                
                if math.gcd(math.gcd(a, b), N) != 1:
                    continue
                
                if check_girth_fast(N, a, b):
                    edges = []
                    for i in range(N):
                        edges.append((i, (i + a) % N))
                        edges.append((i, (i + b) % N))
                    return N, edges
        if found:
            break

    return luw_construct(k, g)
