import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # We need girth >= g.
    # Girth of CD(d, q) is d + 5 if d is odd, d + 4 if d is even?
    # Prompt says: "girth is at least d + 5 for odd d".
    # For g=9, we need d+5 >= 9 => d >= 4.
    # If d=4 (even), girth is at least 8? Or 9?
    # Let's assume we need d such that girth >= g.
    # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: g-5=4 (even) => d = 9-4=5.
    # So d=5.
    # n = 2 * q^d = 2 * 5^5 = 6250.
    # Wait, the seed code:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # If g=9, g-5=4. 4 % 2 == 0. So d = 9 - 4 = 5.
    # n = 2 * 5^5 = 6250.
    # But earlier I thought n=1250 for d=4.
    # If d=4, girth is at least 4+5=9? Or 4+4=8?
    # If d=4 gives girth 9, then n=1250.
    # If d=4 gives girth 8, we need d=5, n=6250.
    # The prompt says "smallest known such graph has 1116 vertices".
    # If the seed produces 6250, the score is 6250/6250 = 1.0?
    # No, the score is seed/value. Seed value is the n produced by the seed.
    # If seed produces 6250, and I produce 1250 (if valid), score = 6250/1250 = 5.0.
    # So I MUST check if d=4 works for g=9.
    
    # Let's check the girth of CD(4, 5).
    # The girth of CD(d, q) is 2d + 2? Or d + 5?
    # Literature: "The girth of CD(d, q) is at least d + 5 for odd d".
    # What about even d?
    # If d=4, is the girth 9?
    # If yes, n=1250 is a huge improvement over 6250.
    
    # Let's try to construct with d=4 and verify girth.
    # If girth < 9, fall back to d=5.
    
    def build_luw(d):
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try d = 4 first for g=9
    if g <= 9:
        n4, e4 = build_luw(4)
        # Verify girth of e4
        # Simple check: count cycles of length < 9
        # This is expensive for n=1250, but we can do a limited BFS
        # Or just trust the math?
        # If d=4, girth is likely 8 or 9.
        # Let's assume d=4 might be valid for g=9.
        # If invalid, the evaluator will score 0.
        # To be safe, I should verify.
        
        # Verification function
        def check_girth(n, edges, g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each vertex to find shortest cycle
            # This is O(n * (n+E)) which is too slow for n=1250?
            # 1250 * 1250 * 5 ~ 10^7. Might be ok in 10s?
            # But we have a deadline.
            
            # Better: Check for cycles of length 3, 4, 5, 6, 7, 8
            # Using matrix powers or BFS with depth limit.
            
            # For now, let's assume d=4 gives girth 9.
            # If not, we can fall back.
            # But we can't fall back if we return.
            # So we must be sure.
            
            # Let's use a heuristic: if d=4 is risky, use d=5.
            # d=5 gives n=6250.
            # d=4 gives n=1250.
            # The difference is huge.
            # I will bet on d=4 being valid for g=9.
            # Actually, the prompt says "girth is at least d+5 for odd d".
            # It doesn't say for even d.
            # But often the bound is d+4 or d+5.
            # If d=4, girth >= 8.
            # If girth is 8, it's invalid for g=9.
            
            # Let's try to verify d=4 quickly.
            # If it fails, return d=5.
            
            # Quick check for 4-cycles in 1250 vertices
            # Number of 4-cycles is (1/8) * trace(A^4)
            # We can compute A^2 and count common neighbors.
            
            A = [[0]*n4 for _ in range(n4)]
            for u, v in e4:
                A[u][v] = 1
                A[v][u] = 1
            
            # Count 4-cycles
            # For each edge (u,v), count common neighbors of u and v.
            # This is O(E * deg^2) = 3125 * 25 = 78125. Very fast.
            
            has_4_cycle = False
            for u, v in e4:
                common = 0
                for w in adj[u]:
                    if A[w][v]:
                        common += 1
                if common > 0:
                    has_4_cycle = True
                    break
            
            if has_4_cycle:
                # d=4 is invalid for g>=5?
                # Actually, if it has 4-cycles, girth <= 4.
                # LUW graphs are usually girth >= 5 or 6.
                # If d=4 has 4-cycles, it's bad.
                pass
            
            # Let's just return d=5 to be safe?
            # But d=5 is 6250.
            # The seed used d=5 (since g-4=5).
            # So the seed score is based on 6250.
            # If I return 1250 and it's valid, I win big.
            # If I return 1250 and it's invalid, I score 0.
            
            # I will return d=5 (6250) to be safe, as it is guaranteed valid by the seed logic.
            # Wait, the seed logic chose d=5.
            # So the seed n is 6250.
            # If I return 6250, score is 1.0.
            
            # Can I find a better one?
            # What if I use a different q?
            # No, q=k=5.
            
            # I will return the seed result (d=5) to ensure validity.
            n5, e5 = build_luw(5)
            return n5, e5
    else:
        # General case
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, e = build_luw(d)
        return n, e

# The above logic is a bit messy. I will clean it up.
# I will stick to the seed's d calculation to guarantee validity.
# But I will add the deadline guard.

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    q = k
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
