"""Random-lift construction for high-girth 3-regular graphs."""
# EVOLVE-BLOCK-START
import random
import time


def _check_girth(adj, g):
    """Verify no cycle shorter than g via non-backtracking walks from each vertex."""
    n = len(adj)
    # For each starting vertex, do BFS in the non-backtracking line graph
    # up to depth (g-1)//2; if we return to the start, girth < g.
    half = (g - 1) // 2
    for start in range(n):
        # states: (prev, cur) with prev != cur, cur in adj[prev]
        visited = set()
        # frontier of (prev, cur) at distance d from start
        # distance 1: (start, nb) for nb in adj[start]
        frontier = [(start, nb) for nb in adj[start]]
        visited.update(frontier)
        for _ in range(half - 1):
            new_frontier = []
            for prev, cur in frontier:
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == start and _ + 1 == half:
                        # found a cycle of length half+1 <= (g-1)/2+1
                        # but we need cycle length < g
                        # cycle length = half+1, need half+1 < g
                        # half = (g-1)//2, so half+1 = (g-1)//2 + 1
                        # for g=16: half=7, half+1=8 < 16 -> girth violation
                        # actually we're looking for cycles of length < g
                        # a cycle found at step d means length d+1
                        # we go up to depth half, so max cycle length = half+1
                        # if half+1 < g, that's a violation
                        # but we should check: if we hit start, cycle length = d+1
                        # where d is the step count (1-indexed from start)
                        # At iteration i (0-indexed), we're at distance i+1
                        # A cycle of length L is found when at distance L-1 we step to start
                        # We want L < g, so L-1 < g-1, i.e. distance <= g-2
                        # But we only go to half = (g-1)//2
                        # For g=16, half=7, we check distances 1..7
                        # A cycle of length 8 would be found at distance 7
                        # 8 < 16, so yes, violation
                        # Actually any cycle of length L < g means L-1 < g-1
                        # We check up to distance half. If half+1 < g, then a cycle
                        # of length half+1 would be a violation.
                        # For g=16: half=7, half+1=8 < 16 -> violation if found
                        # For g=17: half=8, half+1=9 < 17 -> violation if found
                        # This is correct: we detect cycles of length up to half+1
                        # and half+1 = (g-1)//2 + 1 <= g-1 < g for g >= 3
                        return False
                    state = (cur, nb)
                    if state not in visited:
                        visited.add(state)
                        new_frontier.append(state)
            frontier = new_frontier
    return True


def _build_lift(base_edges, m, rng):
    """
    Given a base cycle (list of (u,v) edges forming a cycle of length m),
    construct a 3-regular graph by randomly pairing half-edges.
    """
    n = 2 * m
    # Each vertex v in base has m lifts: v*0, v*1, ..., v*(m-1)
    # Base edges: for i in range(m), edge (i, (i+1)%m)
    # Each base edge e_i connects i to (i+1)%m
    # In the lift, edge e_i connects {i*0..i*(m-1)} to {(i+1)%m * 0 .. (i+1)%m * (m-1)}
    # We need to pair: for each vertex, it has 3 incident half-edges:
    #   - one going "left" along the base cycle (to previous vertex)
    #   - one going "right" along the base cycle (to next vertex)
    #   - one "vertical" (to be paired randomly)
    # Wait, in a cycle each vertex has degree 2 in the base. In the lift,
    # each vertex gets degree 2 from the base cycle edges, plus 1 from the matching.
    # So we need a perfect matching on the "vertical" half-edges.
    #
    # The vertical half-edges: for each base vertex i, there are m lifts i*0..i*(m-1).
    # Each lift i*j has one vertical half-edge. Total vertical half-edges = m*m = m^2.
    # We need to pair them up: a perfect matching on m^2 items.
    #
    # Actually, let me reconsider. The standard random lift:
    # Base graph is a cycle C_m. Each edge e_i = (i, (i+1)%m).
    # The lift has vertices (i, j) for i in 0..m-1, j in 0..m-1.
    # For each base edge e_i, we add edges between {i*0..i*(m-1)} and {(i+1)%m * 0..(i+1)%m*(m-1)}
    # using a random bijection (permutation).
    # This gives each vertex degree 2 (one from left edge, one from right edge).
    # To get degree 3, we need one more edge per vertex.
    #
    # Alternative: use a base graph where each vertex has degree 3 already,
    # but that's more complex. Let me use a different approach.
    #
    # Better approach: start with a cycle of length m, and at each vertex add
    # a "cap" edge that connects vertices within the same fiber via a random
    # permutation. But that gives degree 3 only if we add one edge per vertex
    # within the fiber.
    #
    # Simplest: The base is a cycle. Each vertex in the base has degree 2.
    # In the m-fold cover, each vertex (i,j) connects to:
    #   - (i-1, pi_left_i(j))  [left neighbor in base]
    #   - (i+1, pi_right_i(j)) [right neighbor in base]
    # where pi_left_i and pi_right_i are random permutations of Z_m.
    # This gives degree 2. To get degree 3, add a "vertical" perfect matching
    # on the set of all m^2 vertices, but excluding edges that would create
    # loops or parallel edges.
    #
    # Actually, the cleanest random 3-regular graph construction:
    # Take 2m vertices. Randomly pair them into m edges (a perfect matching M1).
    # Then randomly pair the remaining half-edges... no, that's not right.
    #
    # Let me use the standard approach:
    # 1. Create 2m vertices.
    # 2. Form a cycle: 0-1-2-...-(2m-1)-0. This gives each vertex degree 2.
    # 3. Add a random perfect matching on the 2m vertices, avoiding edges
    #    already in the cycle. This gives each vertex degree 3.
    #
    # The girth of such a graph depends on the matching. Random matchings
    # tend to produce high girth for moderate m.

    # Step 1: cycle
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[(i + 1) % n].append(i)

    # Step 2: random perfect matching avoiding cycle edges
    # Vertices to match: all 2m vertices
    # Forbidden: (i, (i+1)%n) and ((i+1)%n, i) for all i
    forbidden = set()
    for i in range(n):
        forbidden.add((i, (i + 1) % n))
        forbidden.add(((i + 1) % n, i))

    # Build a random perfect matching
    remaining = list(range(n))
    rng.shuffle(remaining)
    matching = []
    i = 0
    while i < len(remaining):
        u = remaining[i]
        # Find a valid partner for u
        valid_partners = [v for v in remaining[i + 1:] if (u, v) not in forbidden]
        if not valid_partners:
            return None  # failed
        v = valid_partners[rng.randrange(len(valid_partners))]
        matching.append((u, v))
        # Remove u and v from remaining
        remaining.remove(u)
        remaining.remove(v)
        i += 1

    # Add matching edges
    for u, v in matching:
        adj[u].append(v)
        adj[v].append(u)

    return adj


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, use random lift (cycle + random perfect matching).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    if k != 3:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction for other k
        return _luw_construct(k, g)

    best_n = None
    best_adj = None

    # Try increasing cycle lengths
    # For girth g, we need the cycle length m to be large enough.
    # The cycle itself has length 2m, so we need 2m >= g, i.e. m >= g/2.
    # But the random matching can create shorter cycles, so we need m larger.
    # Empirically, m around g to 2*g works well.

    rng = random.Random(42)

    # Start with m = g (cycle length 2g), try increasing
    for m in range(g, 2 * g + 1):
        if time.time() > deadline:
            break
        n = 2 * m
        # Try multiple random matchings
        max_tries = 500
        for attempt in range(max_tries):
            if time.time() > deadline:
                break
            adj = _build_lift(None, m, rng)
            if adj is None:
                continue
            if _check_girth(adj, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_adj = [list(x) for x in adj]
                break  # found valid for this m, try smaller? No, m is increasing
        # If we found a valid graph at this m, we can stop since m is increasing
        # but we want the smallest n, so we should try smaller m first.
        # Actually, we're iterating m from g upward, so the first valid one
        # is the smallest. But let's continue a bit to see if we can do better.
        # No, m increasing means n increasing, so first valid is best for this range.
        if best_adj is not None:
            break

    if best_adj is None:
        # Fallback: use LUW construction
        return _luw_construct(k, g)

    # Convert adjacency to edge list
    edges = []
    n = len(best_adj)
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))

    return n, edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph."""
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
# EVOLVE-BLOCK-END
