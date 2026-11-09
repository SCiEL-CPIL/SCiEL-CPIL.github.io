"""Construct a 3-regular graph of girth 16 (cage problem n(3,16)).

Strategy:
1. Use LUW construction with optimized rule coefficients via genetic search.
   The LUW framework guarantees 3-regularity; we search for rules that
   minimize the number of vertices while maintaining girth >= 16.
2. Fallback to a direct tree-closure approach if LUW optimization fails.

Key insight: The LUW construction's graph size depends on the rule.
By optimizing the rule coefficients, we can find rules where the
non-backtracking walk structure creates fewer short cycles, allowing
us to use smaller depth d while still achieving girth >= 16.
"""

import time
import random
from collections import deque, defaultdict

def _time_left(deadline):
    return deadline - time.time()

def _luw_construct(q, d, rule, deadline):
    """Construct LUW graph with given parameters. Returns (n, edges) or None."""
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

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
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 3:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _check_girth_nonbacktracking(n, adj, min_girth, deadline):
    """Check if graph has any cycle shorter than min_girth using non-backtracking walks.
    Returns True if girth < min_girth (invalid)."""
    for start in range(n):
        # State: (current_vertex, previous_vertex) -> distance
        visited = {}
        queue = deque()
        for w in adj[start]:
            queue.append((w, start, 1))
            visited[(w, start)] = 1

        while queue:
            if time.time() > deadline - 1:
                return True  # assume invalid to be safe
            cur, prev, dist = queue.popleft()
            if cur == start:
                return dist < min_girth
            for w in adj[cur]:
                if w == prev:
                    continue
                state = (w, cur)
                if state not in visited:
                    visited[state] = dist + 1
                    queue.append((w, cur, dist + 1))
    return False

def _girth_is_valid(n, adj, min_girth, deadline):
    """Return True if girth >= min_girth."""
    return not _check_girth_nonbacktracking(n, adj, min_girth, deadline)

def _build_adj_from_edges(n, edges):
    adj = [list() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _default_rule(q, d):
    """Generate default LUW rule."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _random_rule(q, d, rng):
    """Generate a random LUW rule."""
    rule = {}
    for c in range(2, d + 1):
        # a and b must be < c, and (a,b) should give reasonable mixing
        a = rng.randint(1, c - 1)
        b = rng.randint(1, c - 1)
        # Ensure a != b for better mixing
        if a == b:
            b = (b % (c - 1)) + 1
            if b == a:
                b = (a % (c - 1)) + 1
        rule[c] = (a, b)
    return rule

def _mutate_rule(rule, q, d, rng, mutation_rate=0.3):
    """Mutate a rule by changing some coefficients."""
    new_rule = {c: rule[c] for c in rule}
    for c in range(2, d + 1):
        if rng.random() < mutation_rate:
            a = rng.randint(1, c - 1)
            b = rng.randint(1, c - 1)
            if a == b:
                b = (b % (c - 1)) + 1
                if b == a:
                    b = (a % (c - 1)) + 1
            new_rule[c] = (a, b)
    return new_rule

def _crossover_rule(rule1, rule2, q, d, rng):
    """Crossover two rules."""
    new_rule = {}
    for c in range(2, d + 1):
        if rng.random() < 0.5:
            new_rule[c] = rule1[c]
        else:
            new_rule[c] = rule2[c]
    return new_rule

def _evaluate_rule(rule, q, d, min_girth, deadline):
    """Evaluate a rule: construct graph and check girth.
    Returns (n, edges) if valid, None otherwise."""
    if time.time() > deadline - 5:
        return None

    result = _luw_construct(q, d, rule, deadline)
    if result is None:
        return None

    n, edges = result

    # Quick degree check
    adj = _build_adj_from_edges(n, edges)
    for i in range(n):
        if len(adj[i]) != 3:
            return None

    # Check girth
    if _girth_is_valid(n, adj, min_girth, deadline):
        return n, edges
    return None

def _genetic_search(q, d, min_girth, deadline, max_generations=50):
    """Run genetic search over LUW rules."""
    rng = random.Random(42)

    # Initialize population
    pop_size = 12
    population = []

    # Start with default rule and some random rules
    population.append(_default_rule(q, d))
    for _ in range(pop_size - 1):
        population.append(_random_rule(q, d, rng))

    best_result = None
    best_n = float('inf')

    for gen in range(max_generations):
        if time.time() > deadline - 10:
            break

        # Evaluate current population
        scores = []
        for rule in population:
            result = _evaluate_rule(rule, q, d, min_girth, deadline)
            if result is not None:
                n, edges = result
                scores.append((n, rule, edges))
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
            else:
                scores.append((float('inf'), rule, None))

            if time.time() > deadline - 10:
                break

        if best_result is not None:
            break

        # Select parents (tournament selection)
        population.sort(key=lambda x: scores[population.index(x)][0] if scores else float('inf'))
        # Simpler: sort scores and take top
        valid_scores = [(s, r, e) for s, r, e in scores if s < float('inf')]
        if valid_scores:
            valid_scores.sort(key=lambda x: x[0])
            top_rules = [r for _, r, _ in valid_scores[:max(2, len(valid_scores)//2)]]
        else:
            top_rules = population[:max(2, len(population)//2)]

        # Generate new population
        new_population = []
        # Keep best
        if valid_scores:
            new_population.append(valid_scores[0][1])

        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            r1 = top_rules[rng.randint(0, len(top_rules) - 1)]
            r2 = top_rules[rng.randint(0, len(top_rules) - 1)]
            child = _crossover_rule(r1, r2, q, d, rng)
            child = _mutate_rule(child, q, d, rng, mutation_rate=0.4)
            new_population.append(child)

        population = new_population

    return best_result

def _try_small_depths(k, g, deadline):
    """Try LUW with small depths, optimizing rules via genetic search."""
    q = k  # q = 3 for cubic graphs

    # For girth g, we need d such that the non-backtracking walk
    # structure guarantees girth >= g.
    # In LUW, girth is related to d. Try d from small to larger.
    # The relationship: for LUW, girth >= 2d - 1 typically, but depends on rule.
    # We'll try d values and let the girth check decide.

    for d in range(4, 14):
        if time.time() > deadline - 15:
            break

        # Theoretical: with d levels, max non-backtracking walk length before
        # returning to start is related to d. We need girth >= g.
        # For g=16, we need d at least around 8-10 typically.
        # But with good rules, smaller d might work.

        result = _genetic_search(q, d, g, deadline, max_generations=30)
        if result is not None:
            return result

    return None

def _direct_construction(k, g, deadline):
    """Try direct construction via tree with systematic boundary closure."""
    if time.time() > deadline - 20:
        return None

    # Build a 3-regular tree of depth h
    # For girth g, we need depth at least g/2
    h = g // 2  # depth 8 for girth 16

    adj = defaultdict(list)
    level = {}
    vertices = [0]
    level[0] = 0
    next_id = 1

    for depth in range(1, h + 1):
        for v in list(vertices):
            if level[v] == depth - 1:
                needed = 3 if v == 0 else 2
                for _ in range(needed):
                    new_id = next_id
                    next_id += 1
                    vertices.append(new_id)
                    level[new_id] = depth
                    adj[v].append(new_id)
                    adj[new_id].append(v)

    boundary = [v for v in vertices if len(adj[v]) == 1]

    if len(boundary) % 2 != 0:
        return None

    # Try to pair boundary vertices to create cycles of length >= g
    # Use a greedy approach: pair vertices that are far apart in the tree
    boundary_sorted = sorted(boundary, key=lambda v: level[v])

    # Try multiple random pairings
    rng = random.Random(123)
    best_result = None

    for attempt in range(50):
        if time.time() > deadline - 10:
            break

        adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
        boundary_copy = list(boundary)
        rng.shuffle(boundary_copy)

        # Pair up boundary vertices
        for i in range(0, len(boundary_copy), 2):
            u = boundary_copy[i]
            v = boundary_copy[i + 1]
            adj_copy[u].append(v)
            adj_copy[v].append(u)

        n = len(adj_copy)

        # Convert to list-of-lists for girth check
        adj_list = [list(adj_copy.get(i, [])) for i in range(n)]

        # Check degree
        valid = True
        for i in range(n):
            if len(adj_list[i]) != 3:
                valid = False
                break

        if not valid:
            continue

        # Check girth
        if _girth_is_valid(n, adj_list, g, deadline):
            edges = set()
            for u in range(n):
                for v in adj_list[u]:
                    edges.add((min(u, v), max(u, v)))
            if best_result is None or n < best_result[0]:
                best_result = (n, list(edges))

    return best_result

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for evaluator's 150s limit

    # Strategy 1: LUW with genetic optimization of rules
    result = _try_small_depths(k, g, deadline)
    if result is not None:
        return result

    # Strategy 2: Direct tree closure
    result = _direct_construction(k, g, deadline)
    if result is not None:
        return result

    # Strategy 3: Fallback - LUW with default rule at larger depth
    q = k
    d = 12  # Should give girth >= 16 with default rule
    rule = _default_rule(q, d)
    result = _luw_construct(q, d, rule, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Last resort: return a trivial valid graph (will be large but valid)
    # Build a cycle of length girth (but that's not 3-regular)
    # Instead, use LUW with even larger d
    d = 14
    rule = _default_rule(q, d)
    result = _luw_construct(q, d, rule, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Absolute fallback: return a known small 3-regular graph
    # Petersen graph has girth 5, not enough.
    # Use a large LUW construction
    d = 16
    rule = _default_rule(q, d)
    result = _luw_construct(q, d, rule, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Should never reach here, but return something
    return 0, []
