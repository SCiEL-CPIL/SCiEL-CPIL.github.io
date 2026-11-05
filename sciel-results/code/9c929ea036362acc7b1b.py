"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _check_girth(n, edges, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    if n == 0:
        return True
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Check degree
    for v in range(n):
        if len(adj[v]) != len(set(adj[v])):
            return False  # multi-edge
    
    # Non-backtracking walk check: for each vertex, do BFS avoiding immediate backtracking
    # A cycle of length L exists iff there's a non-backtracking walk of length L from v back to v
    # We need to check that no non-backtracking closed walk of length < g exists.
    # 
    # Efficient approach: for each vertex v, do a BFS where state is (current, prev)
    # and we track the shortest non-backtracking path back to v.
    
    # For small girth requirements, we can do limited-depth non-backtracking BFS
    max_depth = g - 1  # we need to ensure no cycle shorter than g, so check up to g-1
    
    for start in range(n):
        # State: (current_node, previous_node) -> depth
        # Start: for each neighbor u of start, state (u, start) at depth 1
        visited = set()
        queue = []
        for u in adj[start]:
            state = (u, start)
            if state not in visited:
                visited.add(state)
                queue.append((u, start, 1))
        
        qi = 0
        while qi < len(queue):
            cur, prev, depth = queue[qi]
            qi += 1
            if depth >= max_depth:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue  # no backtracking
                if nxt == start and depth + 1 < g:
                    # Found a cycle of length depth+1 < g
                    return False
                new_state = (nxt, cur)
                if new_state not in visited:
                    visited.add(new_state)
                    if depth + 1 < max_depth:
                        queue.append((nxt, cur, depth + 1))
    return True


def construct(k, g):
    """Construct a k-regular graph with girth >= g using random greedy + local search.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i, j).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Moore bound lower estimate
    if k == 1:
        moore = 2 if g >= 2 else 1
    elif k == 2:
        moore = max(g, 3)  # cycle of length g
    else:
        # Moore bound for (k, g) cage
        if g % 2 == 0:
            # even girth: n >= k * (k-1)^(g/2 - 1)
            moore = k * (k - 1) ** (g // 2 - 1)
        else:
            # odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
            half = (g - 3) // 2
            s = 0
            term = 1
            for i in range(half + 1):
                s += term
                term *= (k - 1)
            moore = 1 + k * s
    
    # Try to construct using random greedy algorithm
    # For the specific case k=5, g=9, we know n=1116 is achievable.
    # We'll use a randomized greedy approach with backtracking.
    
    best_n = None
    best_edges = None
    
    # Strategy: use a random greedy algorithm to build a k-regular graph
    # with large girth. Start with target n near the known best.
    
    # For k=5, g=9: known best is 1116. Let's try to construct near that.
    # General strategy: random greedy edge addition with girth constraint.
    
    # Determine a reasonable starting n
    # Use Moore bound as lower bound, try a few multipliers
    target_ns = []
    if k == 5 and g == 9:
        # Known constructions
        target_ns = [1116, 1100, 1050, 1000, 950, 900, 850, 800, 750, 700]
    else:
        # Try a range around Moore bound
        base = max(moore, 2 * k)
        for mult in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
            target_ns.append(int(base * mult))
        # Sort by n (try smaller first)
        target_ns = sorted(set(target_ns))
    
    random.seed(42)
    
    for target_n in target_ns:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random attempts for this target_n
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            n = target_n
            # Try to build a k-regular graph on n vertices with girth >= g
            success, edges = _random_greedy(n, k, g, deadline)
            if success:
                # Verify
                if _check_girth(n, edges, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    break  # Found valid for this n, try smaller n
                else:
                    # Girth check failed, shouldn't happen if greedy was correct
                    continue
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    return _luw_construct(k, g)


def _random_greedy(n, k, g, deadline):
    """Random greedy algorithm to construct a k-regular graph with girth >= g.
    
    Returns (success, edges).
    """
    # Each vertex needs degree k
    # We'll add edges one at a time, maintaining the girth constraint
    
    # Adjacency sets for quick lookup
    adj = [set() for _ in range(n)]
    edges = []
    
    # Pending edges: for each vertex, how many more edges it needs
    pending = [k] * n
    
    # Non-backtracking check function
    def has_short_cycle(v, u, depth):
        """Check if adding edge (v, u) creates a cycle shorter than g.
        We do a non-backtracking BFS from both endpoints to see if they're connected
        by a non-backtracking path of length < g - 1 (since the new edge adds 1)."""
        if depth >= g - 1:
            return False
        
        # BFS from v avoiding u (the edge we're about to add)
        # Actually, we need to check if v and u are connected by a path of length < g-1
        # that doesn't use the edge (v,u) itself.
        
        # Simpler: check if there's a non-backtracking walk from v to u of length < g-1
        # (not using edge v-u)
        
        # BFS with state (node, prev)
        visited = set()
        queue = [(v, -1, 0)]
        qi = 0
        while qi < len(queue):
            cur, prev, d = queue[qi]
            qi += 1
            if d >= g - 1:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == u and d + 1 < g - 1:
                    return True
                if (nxt, cur) not in visited:
                    visited.add((nxt, cur))
                    if d + 1 < g - 1:
                        queue.append((nxt, cur, d + 1))
        return False
    
    # Greedy: repeatedly pick a random pair of vertices with pending > 0
    # and add the edge if it doesn't create a short cycle
    remaining = n * k // 2  # total edges to add
    attempts = 0
    max_attempts = remaining * 20
    
    while remaining > 0 and attempts < max_attempts:
        if time.time() > deadline - 5:
            return False, None
        attempts += 1
        
        # Pick random vertex with pending > 0
        candidates = [i for i in range(n) if pending[i] > 0]
        if not candidates:
            break
        v = random.choice(candidates)
        
        # Pick random other vertex with pending > 0
        others = [j for j in range(n) if pending[j] > 0 and j != v and j not in adj[v]]
        if not others:
            # Can't complete, fail
            return False, None
        
        # Try a few random choices
        random.shuffle(others)
        added = False
        for u in others[:10]:  # try up to 10 random choices
            if has_short_cycle(v, u, 0):
                continue
            # Add edge
            adj[v].add(u)
            adj[u].add(v)
            pending[v] -= 1
            pending[u] -= 1
            edges.append((v, u))
            remaining -= 1
            added = True
            break
        
        if not added:
            # Try more broadly or fail
            # If we can't find any valid edge, we might be stuck
            # Try a few more times
            success = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    return False, None
                v2 = random.choice(candidates)
                others2 = [j for j in range(n) if pending[j] > 0 and j != v2 and j not in adj[v2]]
                if not others2:
                    continue
                random.shuffle(others2)
                for u2 in others2[:5]:
                    if not has_short_cycle(v2, u2, 0):
                        adj[v2].add(u2)
                        adj[u2].add(v2)
                        pending[v2] -= 1
                        pending[u2] -= 1
                        edges.append((v2, u2))
                        remaining -= 1
                        success = True
                        break
                if success:
                    break
            if not success:
                return False, None
    
    if remaining > 0:
        return False, None
    
    # Check all vertices have degree k
    for i in range(n):
        if len(adj[i]) != k:
            return False, None
    
    return True, edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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
