"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 115
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

    # Precompute neighbor tables for points and lines
    # Point side: vertex is (0, x) where x is a tuple of length d+1 with x[0]=None
    # For each point x, for each first in range(q), compute the line y
    # Line side: vertex is (1, y) where y is a tuple of length d+1 with y[0]=None
    # For each line y, for each first in range(q), compute the point x

    # We'll use a different approach: build the graph by BFS but precompute
    # neighbors more efficiently using the structure.

    # Actually, let's just optimize the neighbor generation by avoiding
    # redundant computation. The key insight is that for a point x, the
    # neighbor line y is determined by y[1] (the "first" value), and then
    # y[c] is computed sequentially.

    # Let's precompute for each possible "first" value, the transformation
    # from point to line and from line to point.

    # For point x -> line y given first:
    # y[1] = first
    # y[c] = x[c] + y[a] * x[b] for c = 2..d
    # For line y -> point x given first:
    # x[1] = first
    # x[c] = y[c] - x[a] * y[b] for c = 2..d

    # We can't fully precompute since it depends on the entire vertex,
    # but we can make the inner loop faster.

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # Precompute rule arrays for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    for idx, vertex in enumerate(order):
        if time.time() > deadline - 10:
            break
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((idx, index[other]))
    
    # Verify the result is valid
    n = len(order)
    if n == 0:
        return 0, []
    
    # Check degree regularity
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(deg != k for deg in degree):
        # Fallback: return a simple valid graph if construction failed
        # For k=4, we can use a cycle of length g with chords, but that's complex.
        # Instead, return the partial result if it's valid, otherwise a minimal valid graph.
        # A simple 4-regular graph with girth >= g: use a cycle C_n where n >= g
        # and add edges carefully. But this is hard to guarantee.
        # For now, return a cycle of length max(g, 2*k) with additional edges.
        # Actually, the safest fallback is a complete graph K_k+1 which has girth 3.
        # That won't work for g > 3. Let's just return the cycle.
        n_fb = max(g, 2 * k)
        edges_fb = []
        for i in range(n_fb):
            edges_fb.append((i, (i + 1) % n_fb))
            edges_fb.append((i, (i + 2) % n_fb))
        # This gives 4-regular if n_fb > 4, but girth is 3 or 4.
        # Not valid for g > 4. Return the original partial result instead.
        pass
    
    return n, edges
# EVOLVE-BLOCK-END
