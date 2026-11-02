import time
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # If k and g don't match the specific case, use a generic fallback
    # The task is specific to k=4, g=10.
    if k != 4 or g != 10:
        # Fallback to LUW for other cases, though task implies k=4, g=10
        # Re-implementing LUW briefly for safety if k,g differ
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

    # Specific search for k=4, g=10
    # Moore bound is 242. Best known is 320.
    # We search for Circulant Graphs C_n(a, b) with girth 10.
    
    def has_short_cycle(n, a, b):
        # Check for cycles of length 2, 4, 6, 8
        # Steps are +/-a, +/-b
        steps = [a, -a, b, -b]
        
        # Length 2
        for s in steps:
            if (2 * s) % n == 0:
                return True
        if (a + b) % n == 0 or (a - b) % n == 0:
            return True
            
        # Length 4
        # Combinations of 4 steps summing to 0
        # a+a+a+a, a+a+b+b, a+b+a+b, a+b+b+a, b+b+b+b, a+a+b-a (0), etc.
        # Non-trivial:
        if (4 * a) % n == 0: return True
        if (4 * b) % n == 0: return True
        if (2 * a + 2 * b) % n == 0: return True
        if (2 * a - 2 * b) % n == 0: return True
        if (a + b + a - b) % n == 0: return True # 0, trivial
        
        # Length 6
        # a+a+a+a+a+a, a+a+a+a+b+b, ...
        # Coefficients for a, b must sum to 0 mod n
        # Possible coeffs for a: -6 to 6, same for b.
        # Total steps 6.
        # We can just do a BFS of depth 3 from 0?
        # If we return to 0 in 6 steps, we have a cycle.
        # But we must ensure it's a simple cycle (no repeated vertices).
        # For girth check, if there is ANY cycle of length < 10, it fails.
        # A closed walk of length L < 10 might not be a simple cycle,
        # but if it is a closed walk, it contains a cycle of length <= L.
        # So if we find a closed walk of length 2, 4, 6, 8, we fail.
        
        # BFS for closed walks of length 2, 4, 6, 8
        # Actually, just checking linear combinations is faster.
        # For length 6:
        # 6a, 5a+b, 5a-b, 4a+2b, 4a-2b, 4a, 3a+3b, 3a-3b, 2a+4b, ...
        # We iterate over possible counts of a and b.
        # Let k_a be number of +a, l_a be number of -a, etc.
        # This is complex.
        
        # Simple BFS for short cycles
        # Start at 0.
        # Depth 1: {a, -a, b, -b}
        # Depth 2: {2a, 0, a+b, a-b, -2a, ...}
        # If 0 appears at depth 2, 4, 6, 8, we have a closed walk.
        # We need to ensure the path is simple?
        # If a closed walk of length L exists, there is a cycle of length <= L.
        # So if we reach 0 at depth 2, 4, 6, 8, we have a cycle of length <= 2, 4, 6, 8.
        # Thus, girth < 10.
        
        visited = {0}
        current = {0}
        for depth in range(1, 9):
            next_frontier = set()
            for node in current:
                for s in steps:
                    new_node = (node + s) % n
                    if new_node == 0 and depth < 10:
                        # Found a closed walk of length depth
                        # Is it a simple cycle?
                        # If depth is 2, 4, 6, 8, it implies a cycle of that length or smaller.
                        # For example, depth 2: a + (-a) = 0. Cycle of length 2.
                        # Depth 4: a+a+a+(-a) = a. Not 0.
                        # a+a+b+(-b) = 2a.
                        # a+b+(-a)+(-b) = 0. Cycle of length 4?
                        # Path: 0 -> a -> a+b -> b -> 0.
                        # Vertices: 0, a, a+b, b.
                        # If a, b, a+b are distinct and non-zero, it is a simple cycle of length 4.
                        # If they are not distinct, there is a shorter cycle.
                        # So if we reach 0 at depth < 10, girth < 10.
                        return True
                    if new_node not in visited:
                        next_frontier.add(new_node)
            if not next_frontier:
                break
            visited.update(next_frontier)
            current = next_frontier
            
        return False

    best_n = None
    best_edges = None
    
    # Search n from 242 to 320
    for n in range(242, 321):
        if time.time() > deadline - 10:
            break
            
        # Symmetry: a < b, a, b > 0, a, b < n/2
        # Also a, b should be coprime to n for connectivity?
        # If gcd(a, n) = d > 1, the graph is a union of d components.
        # The girth of the graph is the girth of the components.
        # We want a 4-regular graph on n vertices.
        # If it's disconnected, it's still a 4-regular graph on n vertices.
        # But "cage" usually implies connected.
        # Let's assume we want connected.
        
        for a in range(1, n // 2 + 1):
            if math.gcd(a, n) != 1:
                continue
            for b in range(a + 1, n // 2 + 1):
                if math.gcd(b, n) != 1:
                    continue
                
                if time.time() > deadline - 5:
                    break
                
                if not has_short_cycle(n, a, b):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    # Remove duplicates? (i, j) and (j, i)
                    # The evaluator expects a list of edges.
                    # Usually (i, j) with i < j.
                    unique_edges = set()
                    for u, v in edges:
                        if u < v:
                            unique_edges.add((u, v))
                        else:
                            unique_edges.add((v, u))
                    best_n = n
                    best_edges = list(unique_edges)
                    # Return immediately if we find a small one?
                    # We want the smallest n.
                    # So we can return immediately if we find one for the current n.
                    return n, best_edges
                    
        if time.time() > deadline - 5:
            break

    # If no circulant graph found, fall back to LUW
    # (Re-use the LUW code from above or a simplified version)
    # For k=4, g=10, LUW q=4, d=5.
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
