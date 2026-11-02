import time
import random
import itertools
from collections import deque

def construct(k, g):
        start_time = time.time()
        deadline = start_time + 115.0  # 5s safety margin

        # Fallback: LUW graph
        def luw_construct():
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
            i = 0
            while i < len(order):
                if time.time() > deadline:
                    break
                vertex = order[i]
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
                i += 1
            n = len(order)
            return n, edges

        # Try Circulant Search
        # k=5, g=9
        if k == 5 and g == 9:
            def girth_ok(n, s1, s2):
                """Check if circulant graph C(n; s1, s2) has girth >= 9.
                Steps: {s1, -s1, s2, -s2, n//2}
                Need: no non-empty sequence of steps summing to 0 mod n with length < 9,
                where no proper prefix sums to 0.
                """
                if n % 2 != 0:
                    return False
                h = n // 2
                steps = (s1, -s1, s2, -s2, h)
                
                # Check L=3: all 5^3 = 125 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                
                # Check L=4: all 5^4 = 625 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                            for d in range(5):
                                w = (z + steps[d]) % n
                                if w == 0:
                                    return False
                
                # Check L=5: all 5^5 = 3125 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                            for d in range(5):
                                w = (z + steps[d]) % n
                                if w == 0:
                                    return False
                                for e in range(5):
                                    v = (w + steps[e]) % n
                                    if v == 0:
                                        return False
                
                # Check L=6: all 5^6 = 15625 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                            for d in range(5):
                                w = (z + steps[d]) % n
                                if w == 0:
                                    return False
                                for e in range(5):
                                    v = (w + steps[e]) % n
                                    if v == 0:
                                        return False
                                    for f in range(5):
                                        u = (v + steps[f]) % n
                                        if u == 0:
                                            return False
                
                # Check L=7: all 5^7 = 78125 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                            for d in range(5):
                                w = (z + steps[d]) % n
                                if w == 0:
                                    return False
                                for e in range(5):
                                    v = (w + steps[e]) % n
                                    if v == 0:
                                        return False
                                    for f in range(5):
                                        u = (v + steps[f]) % n
                                        if u == 0:
                                            return False
                                        for g_ in range(5):
                                            t = (u + steps[g_]) % n
                                            if t == 0:
                                                return False
                
                # Check L=8: all 5^8 = 390625 combos
                for a in range(5):
                    x = steps[a]
                    for b in range(5):
                        y = (x + steps[b]) % n
                        if y == 0:
                            return False
                        for c in range(5):
                            z = (y + steps[c]) % n
                            if z == 0:
                                return False
                            for d in range(5):
                                w = (z + steps[d]) % n
                                if w == 0:
                                    return False
                                for e in range(5):
                                    v = (w + steps[e]) % n
                                    if v == 0:
                                        return False
                                    for f in range(5):
                                        u = (v + steps[f]) % n
                                        if u == 0:
                                            return False
                                        for g_ in range(5):
                                            t = (u + steps[g_]) % n
                                            if t == 0:
                                                return False
                                            for h_ in range(5):
                                                s = (t + steps[h_]) % n
                                                if s == 0:
                                                    return False
                
                return True

            def make_circulant_edges(n, s1, s2):
                h = n // 2
                steps = (s1, -s1, s2, -s2, h)
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return edges

            # Search for circulant graphs
            # Moore bound for k=5, g=9: 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
            # n must be even (since n/2 is a step)
            # Try n from 428 upward, with limited s1, s2 pairs per n
            
            best_result = None
            best_n = float('inf')
            
            # Limit the search space
            # For each n, try a few (s1, s2) pairs
            # s1, s2 should be relatively small and distinct
            # Also s1, s2, n/2 should be distinct and non-zero
            
            n_start = 428
            n_end = min(500, 428 + 72)  # Try up to ~70 n values
            
            for n in range(n_start, n_end + 1, 2):
                if time.time() > deadline - 5:
                    break
                h = n // 2
                # Try small s1, s2 values
                # s1, s2 should be in range [1, h-1], s1 != s2, s1 != h, s2 != h
                # For speed, try s1 in a small range, s2 in a small range
                s1_candidates = list(range(1, min(20, h)))
                s2_candidates = list(range(1, min(20, h)))
                
                for s1 in s1_candidates:
                    if time.time() > deadline - 5:
                        break
                    for s2 in s2_candidates:
                        if s1 == s2:
                            continue
                        if s1 == h or s2 == h:
                            continue
                        if time.time() > deadline - 5:
                            break
                        if girth_ok(n, s1, s2):
                            if n < best_n:
                                best_n = n
                                best_result = (n, make_circulant_edges(n, s1, s2))
                            break
                    if best_result and best_n <= n:
                        break
                if best_result and best_n <= n:
                    break
            
            if best_result:
                return best_result
            
            # Fallback to LUW
            return luw_construct()
            
        return luw_construct()
