"""Structural redesign: random 4-regular graph + non-backtracking local search (2-switches)
to eliminate cycles shorter than g, with a time-budgeted deadline guard."""
import time
import random
import numpy as np
from collections import defaultdict, deque

_DEADLINE = 115.0  # hard wall-clock limit (evaluator kills at 150s)


def _is_valid_4reg(adj, n):
    """Check adj is a simple 4-regular graph on n vertices."""
    if len(adj) != n:
        return False
    for i in range(n):
        nbrs = adj[i]
        if len(nbrs) != 4:
            return False
        if len(set(nbrs)) != 4:
            return False
        if i in nbrs:
            return False
        for j in nbrs:
            if i not in adj[j]:
                return False
    return True


def _short_cycle_count(adj, n, g):
    """Count non-backtracking closed walks of length < g starting at each vertex.
    Returns total count (0 means girth >= g). Uses BFS on directed edges."""
    # directed edges: (u, v) means u->v
    total = 0
    for start in range(n):
        # state: (current, prev) ; prev = vertex we came from (None for start)
        # We do a DFS/BFS over non-backtracking walks up to length g-1
        # For each start vertex, count closed walks of length L in [3, g-1]
        # A walk v0->v1->...->vL with vL = v0, no backtracking (v_{i+1} != v_{i-2})
        # We count each closed walk once per starting vertex (start fixed).
        # Use iterative deepening; for small g (10) this is cheap.
        # Represent state as (cur, prev) with prev in -1..n-1
        # We want walks of length L ending back at start.
        # Do BFS layer by layer.
        # layer 0: just start (length 0), but we need length >= 3
        # Actually non-backtracking closed walk of length L: v0, v1, ..., vL=v0,
        # with v_{i+1} != v_{i-2} for all i (indices mod L). For L>=3.
        # We'll do a DFS from start, tracking (cur, prev), depth.
        # Count when depth >= 3 and cur == start.
        # But we must avoid infinite loops; max depth = g-1.
        # To count, we do a tree search but prune: since girth is what we're measuring,
        # and for a 4-regular graph the number of non-backtracking walks of length L
        # from a vertex is at most 4*3^(L-1), which for L=9 is 4*3^8 = 78732.
        # That's too many per vertex for 243 vertices. 
        # Better: use the standard girth-checking via BFS on the directed edge graph.
        # For girth checking, we just need to know if ANY non-backtracking cycle of
        # length < g exists. We can do a BFS from each directed edge.
        pass
    # Use a more efficient approach: BFS on directed edges to find shortest non-backtracking cycle.
    # For each directed edge (u,v), do BFS in the directed-edge graph where (u,v)->(v,w) for w != u.
    # A non-backtracking cycle of length L corresponds to a path of L directed edges that returns to start.
    # We want the shortest such cycle.
    # This is the standard algorithm for girth in terms of non-backtracking walks.
    # We'll compute the minimum cycle length over all directed edges.
    min_len = g  # assume no cycle shorter than g
    for u in range(n):
        for v in adj[u]:
            # BFS from directed edge (u, v)
            # state: (cur, prev) = (v, u) initially, length 1
            # We want to reach (u, x) for some x, i.e., cur == u, which closes the cycle.
            # Actually, a non-backtracking walk starting with u->v, after L steps, 
            # we're at some vertex with prev. The cycle closes when we return to u.
            # The cycle length is L+1 (L steps after the first edge, plus the first edge).
            # Wait: directed edge (u,v) is step 1. After L more steps (total L+1 edges),
            # if we're back at u, the cycle has length L+1.
            # BFS: state (cur, prev), we expand to (w, cur) for w in adj[cur], w != prev.
            # Start: (v, u) at depth 1 (one edge used).
            # We want to reach any state (u, prev) => cycle length = depth + 1.
            # But we must not use backtracking: w != prev is enforced.
            visited = set()
            queue = deque()
            queue.append((v, u, 1))  # (cur, prev, edges_used)
            visited.add((v, u))
            found = False
            while queue and not found:
                cur, prev, d = queue.popleft()
                if cur == u and d >= 3:
                    # Cycle of length d + 1? 
                    # We used d edges to get from u to cur (=u), so cycle length = d.
                    # Wait, let me recount. Start edge u->v is edge 1, we're at v with prev=u, d=1.
                    # After expanding, we add edges. When we pop (cur, prev, d), we've used d edges.
                    # If cur == u, we've returned to start, cycle length = d.
                    # But we need d >= 3 (no 2-cycles in simple graph, and no self-loops).
                    if d < min_len:
                        min_len = d
                    found = True  # we found a cycle from this start; but there might be shorter from other starts
                    # Actually we should continue to find the shortest, but BFS guarantees first hit is shortest for this start.
                    # We can break for this start.
                    break
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if (w, cur) not in visited:
                        visited.add((w, cur))
                        queue.append((w, cur, d + 1))
    return min_len if min_len < g else 0


def _find_short_cycles(adj, n, g, max_cycles=50):
    """Find up to max_cycles non-backtracking cycles of length < g.
    Returns list of cycles, each a list of vertices [v0, v1, ..., v_{L-1}] with L < g."""
    cycles = []
    for u in range(n):
        for v in adj[u]:
            # BFS from (u, v)
            # We'll store parent pointers to reconstruct the cycle.
            visited = {}  # (cur, prev) -> (parent_cur, parent_prev, depth)
            queue = deque()
            queue.append((v, u))
            visited[(v, u)] = None  # root
            found_cycle = None
            while queue and found_cycle is None:
                cur, prev = queue.popleft()
                d = len(visited[(cur, prev)]) + 1 if visited[(cur, prev)] is not None else 1
                # Actually let's track depth separately
                # Redo with depth tracking
                pass
            # Simpler: do a DFS with explicit stack and parent tracking
            # Let's do BFS with parent pointers
            visited = {}
            queue = deque()
            queue.append((v, u, 0))  # (cur, prev, depth)
            visited[(v, u)] = 0
            while queue:
                cur, prev, d = queue.popleft()
                if cur == u and d >= 2:
                    # Cycle of length d + 1
                    # Reconstruct: follow parent pointers
                    # We need to store parent. Let's redo.
                    pass
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if (w, cur) not in visited:
                        visited[(w, cur)] = d + 1
                        queue.append((w, cur, d + 1))
    return cycles


def _girth_check(adj, n, g):
    """Return the girth (shortest cycle length) of the graph, or g if no cycle < g.
    Uses NumPy-vectorized BFS on directed edges for speed."""
    # Build adjacency as numpy array for fast indexing
    adj_arr = np.array(adj, dtype=np.int32)
    min_len = g
    for u in range(n):
        for v in adj[u]:
            # BFS from directed edge (u, v)
            # state: (cur, prev), start (v, u)
            # We want to reach (u, *) to close the cycle
            visited = set()
            queue = deque()
            queue.append((v, u, 1))
            visited.add((v, u))
            while queue:
                cur, prev, d = queue.popleft()
                if cur == u:
                    # Cycle of length d
                    if d < min_len:
                        min_len = d
                    break
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if (w, cur) not in visited:
                        visited.add((w, cur))
                        queue.append((w, cur, d + 1))
    return min_len


def _random_4reg(n, rng):
    """Generate a random 4-regular graph on n vertices using Fao's algorithm."""
    # Fao's algorithm for random d-regular graphs
    # n must be even and d*n must be even (4*n is always even)
    # Step 1: create a random permutation of the half-edges
    # Each vertex has 4 half-edges
    # Total half-edges = 4n, pair them up randomly
    half_edges = []
    for i in range(n):
        for j in range(4):
            half_edges.append((i, j))
    rng.shuffle(half_edges)
    # Pair them up
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(half_edges), 2):
        u, _ = half_edges[i]
        v, _ = half_edges[i + 1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    if not valid:
        return None
    # Convert to list
    adj_list = [sorted(adj[i]) for i in range(n)]
    return adj_list


def _two_switch(adj, n, edge1, edge2):
    """Apply a 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c).
    Returns new adj if valid, else None."""
    a, b = edge1
    c, d = edge2
    # Check edges exist
    if b not in adj[a] or d not in adj[c]:
        return None
    # Option 1: add (a,c) and (b,d)
    # Check they don't already exist and are not self-loops
    if a != c and c not in adj[a] and b != d and d not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(c)
        new_adj[c].append(a)
        new_adj[b].append(d)
        new_adj[d].append(b)
        return new_adj
    # Option 2: add (a,d) and (b,c)
    if a != d and d not in adj[a] and b != c and c not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(d)
        new_adj[d].append(a)
        new_adj[b].append(c)
        new_adj[c].append(b)
        return new_adj
    return None


def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # safety margin
    
    if k != 4:
        # Fallback to L-U-W for other k
        return _luw_fallback(k, g)
    
    rng = random.Random(42)
    
    # Moore bound for 4-regular, girth 10:
    # n >= 1 + 4 * sum_{i=0}^{4} 3^i = 1 + 4*(1+3+9+27+81) = 1 + 4*121 = 485? 
    # Wait, for d-regular, girth g (even), Moore bound: n >= 1 + d * sum_{i=0}^{g/2-1} (d-1)^i
    # For d=4, g=10: n >= 1 + 4*(1 + 3 + 9 + 27 + 81) = 1 + 4*121 = 485
    # But the task says best lower bound is 243. Let me recheck.
    # Actually for girth g (even), the Moore bound is:
    # n >= 1 + d * sum_{i=0}^{g/2-1} (d-1)^i
    # For g=10, g/2-1 = 4, so sum_{i=0}^{4} 3^i = 1+3+9+27+81 = 121
    # n >= 1 + 4*121 = 485
    # But the task says 243. Maybe I'm misremembering the formula.
    # Let me just try various n values starting from a reasonable lower bound.
    
    # The task says best lower bound is 243, smallest known is 320.
    # Let's try to find a graph with n as small as possible.
    # Start with n=243 and go up if needed.
    
    best_n = None
    best_edges = None
    
    # Try a range of n values, starting from known feasible sizes
    # The smallest known is 320, so focus effort there and nearby
    n_list = list(range(318, 321)) + list(range(322, 331, 2)) + list(range(332, 351, 2))
    for n in n_list:
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue  # 4-regular requires even n
        # Try multiple random restarts with increasing budget for smaller n
        max_attempts = 5 if n <= 322 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            adj = _random_4reg(n, rng)
            if adj is None:
                continue
            # Check girth
            girth = _girth_check(adj, n, g)
            if girth >= g:
                # Found a valid graph!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break
            else:
                # Try local search to improve girth
                adj = _local_search(adj, n, g, rng, deadline)
                if adj is not None:
                    girth = _girth_check(adj, n, g)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                        break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return a valid graph from L-U-W
    return _luw_fallback(4, g)


def _find_short_cycle(adj, n, g):
    """Find one non-backtracking cycle of length < g. Returns list of vertices or None."""
    for u in range(n):
        for v in adj[u]:
            # BFS from (u, v) with parent tracking
            parent = {}  # (cur, prev) -> (parent_cur, parent_prev)
            depth = {}
            queue = deque()
            queue.append((v, u))
            parent[(v, u)] = None
            depth[(v, u)] = 1
            while queue:
                cur, prev = queue.popleft()
                d = depth[(cur, prev)]
                if cur == u and d >= 3:
                    # Reconstruct cycle
                    cycle = []
                    state = (cur, prev)
                    while state is not None:
                        cycle.append(state[0])
                        state = parent[state]
                    cycle.reverse()
                    # Remove the duplicate start vertex at the end
                    if len(cycle) > 1 and cycle[0] == cycle[-1]:
                        cycle = cycle[:-1]
                    if len(cycle) < g:
                        return cycle
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if (w, cur) not in parent:
                        parent[(w, cur)] = (cur, prev)
                        depth[(w, cur)] = d + 1
                        queue.append((w, cur))
    return None


def _local_search(adj, n, g, rng, deadline):
    """Local search: apply 2-switches to break short cycles."""
    adj = [list(x) for x in adj]
    for iteration in range(5000):
        if time.time() > deadline:
            break
        girth = _girth_check(adj, n, g)
        if girth >= g:
            return adj
        # Find a short cycle and try to break it
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return adj  # No short cycle found (shouldn't happen if girth < g)
        
        # Get all edges
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        if len(edges) < 2:
            break
        
        # Try to break the cycle: pick an edge on the cycle and a random other edge
        cycle_edges = []
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            if a < b:
                cycle_edges.append((a, b))
            else:
                cycle_edges.append((b, a))
        
        # Try multiple attempts to break this cycle
        found_improvement = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            e1 = rng.choice(cycle_edges)
            e2 = rng.choice(edges)
            if e1 == e2:
                continue
            new_adj = _two_switch(adj, n, e1, e2)
            if new_adj is None:
                continue
            new_girth = _girth_check(new_adj, n, g)
            if new_girth > girth:
                adj = new_adj
                girth = new_girth
                found_improvement = True
                break
        
        if not found_improvement:
            # Try random switches as fallback
            for attempt in range(5):
                if time.time() > deadline:
                    break
                e1 = rng.choice(edges)
                e2 = rng.choice(edges)
                if e1 == e2:
                    continue
                new_adj = _two_switch(adj, n, e1, e2)
                if new_adj is None:
                    continue
                new_girth = _girth_check(new_adj, n, g)
                if new_girth > girth:
                    adj = new_adj
                    girth = new_girth
                    break
    return adj


def _luw_fallback(k, g):
    """Fallback to Lazebnik-Ustimenko-Woldar construction."""
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
