import time
import math
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to check if a number is a prime power
    def is_prime_power(n):
        if n < 2: return False
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
        return n == 2 or n == 3 or n == 5 or n == 7 or n == 11 or n == 13 or n == 17 or n == 19 or n == 23 or n == 29 or n == 31
    
    # Specific optimal cases
    if k == 4 and g == 10:
        # Use a known construction or return a valid graph.
        # The LUW construction for k=4, g=10 yields n=1024.
        # We can try to find a smaller one, but for now, let's stick with the LUW fallback 
        # unless we have a specific hardcoded graph.
        # Actually, the task says "Smallest known such graph has 320 vertices".
        # Generating a 320-vertex graph from scratch is hard.
        # Let's just use the LUW construction which is valid but larger.
        # However, we can try to improve the LUW parameters or use a different construction.
        # For now, pass to the LUW logic below.
        pass

    if k == 3 and g == 5:
        # Petersen Graph
        n = 10
        edges = []
        # Outer pentagon 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i+1)%5))
        # Inner star 5-7-9-6-8-5
        inner = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner[i], inner[(i+1)%5]))
        # Spokes
        for i in range(5):
            edges.append((i, inner[i]))
        return n, edges

    if k == 3 and g == 6:
        # Heawood Graph (Incidence of PG(2,2))
        n = 14
        edges = []
        # Points: 0-6, Lines: 7-13
        # PG(2,2) points: (0,0,1),(0,1,1),(1,0,1),(1,1,1),(0,0,1)? No, 7 points.
        # Let's use a known edge list for Heawood
        # Or construct it.
        # Actually, simpler:
        # 0-1-2-0 (triangle? No, girth 6)
        # Heawood is the incidence graph of Fano plane.
        # Fano plane points: 1..7, lines: 1..7
        # Lines: {1,2,3}, {1,4,5}, {1,6,7}, {2,4,6}, {2,5,7}, {3,4,7}, {3,5,6}
        # Vertices: P0..P6, L0..L6
        # Edges: P_i - L_j if i in line j
        fano_lines = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [1, 3, 5],
            [1, 4, 6],
            [2, 3, 6],
            [2, 4, 5]
        ]
        for j, line in enumerate(fano_lines):
            for i in line:
                edges.append((i, 7 + j))
        return n, edges

    if k == 3 and g == 8:
        # Tutte-Coxeter Graph
        n = 30
        # This is a specific graph.
        # We can use a known construction or edge list.
        # For now, fallback to LUW if we don't have it hardcoded, but let's try to construct or return a known one.
        # Actually, the LUW construction for k=3, g=8 (d=3) should give 30 vertices?
        # LUW CD(3,3): q=3. Points/Lines in GF(3)^3?
        # Let's rely on LUW for this if not hardcoded, but LUW might be larger.
        # Tutte-Coxeter is optimal.
        # Hardcoding 30 vertices is tedious. Let's use LUW and hope it finds 30, or accept larger.
        # But we want to beat the seed. If seed uses LUW and gets >30, we win by hardcoding.
        # Let's generate Tutte-Coxeter via a known method or just use LUW and see.
        # Actually, I will implement a generic LUW and use it. If k=3, g=8, LUW(3,3) has 2 * 3^3 = 54 vertices?
        # Wait, LUW CD(d,q) has 2q^d vertices?
        # For d=3, q=3, n=54. Tutte-Coxeter is 30.
        # So hardcoding is better.
        pass # Will be handled by LUW fallback or specific check

    if k == 5 and g == 5:
        # Hoffman-Singleton
        n = 50
        # Hardcode or construct?
        # It's a specific graph.
        pass

    if g == 6:
        q = k - 1
        if is_prime_power(q):
            # Construct Incidence Graph of PG(2, q)
            # Points: (x, y, 1), (x, 1, 0), (1, 0, 0)
            points = []
            for x in range(q):
                for y in range(q):
                    points.append((x, y, 1))
            for x in range(q):
                points.append((x, 1, 0))
            points.append((1, 0, 0))
            
            n_points = len(points)
            # Lines are dual to points
            lines = []
            for x in range(q):
                for y in range(q):
                    lines.append((x, y, 1))
            for x in range(q):
                lines.append((x, 1, 0))
            lines.append((1, 0, 0))
            
            n = 2 * n_points
            edges = []
            
            # Incidence: P . L = 0
            for i, p in enumerate(points):
                for j, l in enumerate(lines):
                    if (p[0]*l[0] + p[1]*l[1] + p[2]*l[2]) % q == 0:
                        edges.append((i, n_points + j))
            
            return n, edges

    # Fallback: Lazebnik-Ustimenko-Woldar (LUW) Construction
    # This is the same as the seed, but we ensure it runs.
    q = k
    # d is chosen such that girth is g
    # Girth of CD(d,q) is d+5 if d is odd, d+4 if d is even?
    # Seed logic:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to seed logic for d.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n = len(order)
    return n, edges
