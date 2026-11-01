"""Construct a 3-regular graph of large girth via random lift + local repair, with girth certification."""
import time
import random
import math
from collections import deque

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 140.0  # seconds from start of construct()
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _time_left():
    return _DEADLINE - _elapsed()


def _moore_bound(k, g):
    """Lower bound on n for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: n >= k * ( (k-1)^(g/2 - 1) - 1 ) / (k-2)  ... standard Moore bound
        # For k=3, g even: n >= 3 * (2^(g/2-1) - 1)  ... let me re-derive
        # Actually for k-regular, girth g (even): n >= k * (k-1)^(g/2 - 1) / (k-2)  is not quite right.
        # Standard: for girth g (even), n >= k * ((k-1)^(g/2 - 1) - 1) / (k - 2)
        # For k=3: n >= 3 * (2^(g/2-1) - 1) / 1 = 3*(2^(g/2-1)-1)
        half = g // 2
        return int(k * ((k - 1) ** (half - 1) - 1) / (k - 2)) if k > 2 else 2
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        # For k=3: n >= 1 + 3*(2^((g-1)/2) - 1)
        half = (g - 1) // 2
        s = sum((k - 1) ** i for i in range(half))
        return 1 + k * s


def _has_short_cycle_bfs(adj, g):
    """Check if graph has any cycle of length < g using BFS from each vertex.
    Returns True if a short cycle exists (bad), False otherwise."""
    n = len(adj)
    for src in range(n):
        # BFS tracking (node, parent, depth)
        visited = {}  # node -> (parent, depth)
        visited[src] = (-1, 0)
        queue = deque([src])
        while queue:
            u = queue.popleft()
            _, depth = visited[u]
            if depth >= g // 2:
                # No cycle shorter than g can be found from here
                # Actually we need depth < g//2 for even, or < (g-1)//2 for odd
                # Let's just check all edges
                continue
            for v in adj[u]:
                if v == visited[u][0]:
                    continue  # don't go back to parent
                if v not in visited:
                    visited[v] = (u, depth + 1)
                    queue.append(v)
                else:
                    # Found a cycle: length = depth + 1 + visited[v][1]
                    cycle_len = depth + 1 + visited[v][1]
                    if cycle_len < g:
                        return True
    return False


def _girth_via_nonbacktracking(adj, g):
    """Certify girth >= g using non-backtracking walks from every vertex.
    Returns True if girth >= g (good), False otherwise."""
    n = len(adj)
    # For each vertex, do BFS on the state space (node, prev_node)
    # A cycle of length < g exists if we find a state (v, u) where we've seen (u, v)
    # at a distance < g.
    for start in range(n):
        # State: (current, previous)
        # Start from each neighbor of start
        dist = {}  # (node, prev) -> distance
        queue = deque()
        for nb in adj[start]:
            state = (nb, start)
            dist[state] = 1
            queue.append(state)
        while queue:
            u, prev = queue.popleft()
            d = dist[(u, prev)]
            if d >= g:
                continue
            for v in adj[u]:
                if v == prev:
                    continue  # non-backtracking
                new_state = (v, u)
                if new_state in dist:
                    # Cycle found: length = d + 1 + dist[new_state]
                    cycle_len = d + 1 + dist[new_state]
                    if cycle_len < g:
                        return False
                else:
                    dist[new_state] = d + 1
                    queue.append(new_state)
    return True


def _generate_random_3_regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list or None if not simple."""
    if n < 4 or n % 2 != 0:
        return None
    # 3n half-edges
    stubs = list(range(n)) + list(range(n)) + list(range(n))
    random.shuffle(stubs)
    # Pair them up
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i in range(0, 3 * n, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None  # multi-edge
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _local_repair(adj, g, max_iters=200):
    """Try to eliminate short cycles by switching edges. Returns (adj, success)."""
    n = len(adj)
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        # Find a short cycle
        short_cycle = _find_one_short_cycle(adj, g)
        if short_cycle is None:
            return adj, True
        # Try to break it by switching
        # Pick a random edge on the cycle and try to switch
        cycle = short_cycle
        if len(cycle) < 3:
            return adj, False
        # Try switching each edge on the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            # Find another neighbor of u and v to switch
            # u has degree 3, one edge to v. Pick another neighbor w of u.
            # v has degree 3, one edge to u. Pick another neighbor x of v.
            # If we remove (u,v) and add (w,x) or (u,x) and (w,v)...
            # Simple switch: pick w != v from adj[u], x != u from adj[v]
            # Remove (u,v), add (w,x) if valid
            candidates_w = [w for w in adj[u] if w != v]
            candidates_x = [x for x in adj[v] if x != u]
            random.shuffle(candidates_w)
            random.shuffle(candidates_x)
            for w in candidates_w:
                for x in candidates_x:
                    if w == x:
                        continue
                    if w == u or x == v:
                        continue
                    # Check if (w,x) is already an edge
                    key_wx = (min(w, x), max(w, x))
                    # Check if w-x edge exists
                    if x in adj[w]:
                        continue
                    # Try the switch: remove (u,v), add (w,x)
                    # But we need to keep degree 3: remove u-v, add w-x
                    # Wait, this changes degrees: u loses v, w gains x. 
                    # That's not a valid 2-switch unless we also handle the other side.
                    # Actually a proper 2-switch: remove (u,v) and (w,x), add (u,x) and (w,v)
                    # Let me reconsider.
                    pass
        # Alternative: just try random edge switches
        # Pick a random edge (a,b), remove it, pick random non-adjacent c,d, add (c,d)
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        random.shuffle(all_edges)
        for a, b in all_edges[:10]:
            # Remove (a,b)
            adj[a].remove(b)
            adj[b].remove(a)
            # Find candidates for new edges
            # Pick two random vertices c,d not adjacent, not a or b
            verts = list(range(n))
            random.shuffle(verts)
            found = False
            for c in verts:
                if c in (a, b):
                    continue
                for d in verts:
                    if d in (a, b) or d == c:
                        continue
                    if d in adj[c]:
                        continue
                    # Check degrees: after removing (a,b), a and b have degree 2
                    # We want to add edges to restore degree 3
                    # Add (a,c) and (b,d) if valid
                    if c in adj[a] or d in adj[b]:
                        continue
                    # Apply
                    adj[a].append(c)
                    adj[b].append(d)
                    adj[c].append(a)
                    adj[d].append(b)
                    found = True
                    break
                if found:
                    break
            if found:
                break
            # Undo removal
            adj[a].append(b)
            adj[b].append(a)
            break
    return adj, _girth_via_nonbacktracking(adj, g)


def _find_one_short_cycle(adj, g):
    """Find one cycle of length < g, return list of vertices or None."""
    n = len(adj)
    for src in range(n):
        visited = {}
        visited[src] = (-1, 0)
        queue = deque([src])
        while queue:
            u = queue.popleft()
            _, depth = visited[u]
            if depth >= g // 2:
                continue
            for v in adj[u]:
                if v == visited[u][0]:
                    continue
                if v not in visited:
                    visited[v] = (u, depth + 1)
                    queue.append(v)
                else:
                    cycle_len = depth + 1 + visited[v][1]
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = []
                        cur = u
                        while cur != v:
                            cycle.append(cur)
                            cur = visited[cur][0]
                        cycle.append(v)
                        # Now go from v back to src
                        cur2 = v
                        while cur2 != src:
                            cycle.append(cur2)
                            cur2 = visited[cur2][0]
                        cycle.append(src)
                        return cycle
    return None


def _try_lift(base_adj, voltage_group_size, g):
    """Try a voltage graph lift. base_adj is a small graph, voltage_group_size is m.
    Returns (n, adj) if a valid graph is found, else None."""
    n = len(base_adj) * voltage_group_size
    # Random voltage assignment
    base_n = len(base_adj)
    voltages = [[random.randint(0, voltage_group_size - 1) for _ in range(len(base_adj[i]))]
                for i in range(base_n)]
    # Build lifted graph
    adj = [[] for _ in range(n)]
    for i in range(base_n):
        for j_idx, j in enumerate(base_adj[i]):
            if i < j:
                v = voltages[i][j_idx]
                # Edge from (i,0) to (j,v) and symmetric
                for offset in range(voltage_group_size):
                    u_node = i * voltage_group_size + offset
                    v_node = j * voltage_group_size + (offset + v) % voltage_group_size
                    # Add edge both ways (but only once per undirected edge)
                    adj[u_node].append(v_node)
                # Also the reverse voltage
                for offset in range(voltage_group_size):
                    u_node = j * voltage_group_size + offset
                    v_node = i * voltage_group_size + (offset - v) % voltage_group_size
                    adj[u_node].append(v_node)
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    # Check girth
    if _girth_via_nonbacktracking(adj, g):
        return n, adj
    return None


def construct(k, g):
    global _start_time
    _start_time = time.time()
    random.seed(42)

    if k != 3:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
        return _luzw_construct(k, g)

    # For k=3, try to build a graph of girth >= g with as few vertices as possible
    moore = _moore_bound(3, g)
    
    best_n = None
    best_edges = None

    # Strategy 1: Try random 3-regular graphs with local repair
    # Start from Moore bound and go up
    # For g=16, Moore bound for k=3 is 3*(2^7 - 1) = 3*127 = 381
    # But we need even number of vertices for 3-regular
    start_n = moore
    if start_n % 2 != 0:
        start_n += 1
    
    # Try a range of sizes
    max_n = 512  # Best known lower bound is 512, smallest known is 936
    # We'll try sizes from start_n to some upper limit
    
    # First, try some known small graphs or random constructions
    for n in range(start_n, min(start_n + 200, max_n + 100), 2):
        if _time_left() < 10.0:
            break
        for attempt in range(5):
            if _time_left() < 5.0:
                break
            adj = _generate_random_3_regular(n)
            if adj is None:
                continue
            # Check girth
            if _girth_via_nonbacktracking(adj, g):
                best_n = n
                # Build edges
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_edges = edges
                return n, edges
        # If we didn't find a valid graph, try local repair on one
        if _time_left() < 10.0:
            break
        adj = _generate_random_3_regular(n)
        if adj is not None:
            adj, success = _local_repair(adj, g, max_iters=50)
            if success:
                best_n = n
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_edges = edges
                return n, edges

    # Strategy 2: Voltage graph lifts from small base graphs
    # Petersen graph (10 vertices, 3-regular)
    petersen_adj = [
        [1, 4, 5],
        [0, 2, 6],
        [1, 3, 7],
        [2, 4, 8],
        [0, 3, 9],
        [4, 6, 8],
        [5, 7, 9],
        [6, 8, 0],
        [7, 9, 1],
        [8, 5, 2]
    ]
    # Wait, let me get the Petersen graph right
    # Petersen: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-(i+5)%5
    petersen_adj = [
        [1, 4, 5],
        [0, 2, 6],
        [1, 3, 7],
        [2, 4, 8],
        [0, 3, 9],
        [0, 7, 8],
        [1, 5, 9],
        [2, 6, 0],
        [3, 7, 1],
        [4, 8, 2]
    ]
    # Let me verify: vertex 0 connects to 1, 4 (pentagon neighbors) and 5 (spoke)
    # Vertex 5 connects to 0 (spoke) and 7, 8 (inner star)
    # Inner star: 5-7-9-6-8-5
    # So 5 connects to 7 and 8. 7 connects to 5 and 9. 9 connects to 7 and 6. 6 connects to 9 and 8. 8 connects to 6 and 5.
    # Let me redo:
    petersen_adj = [
        [1, 4, 5],  # 0
        [0, 2, 6],  # 1
        [1, 3, 7],  # 2
        [2, 4, 8],  # 3
        [0, 3, 9],  # 4
        [0, 7, 8],  # 5
        [1, 9, 0],  # 6  -- wait, 6 should connect to 1 (spoke) and 9, 8 (inner star)
        [2, 5, 9],  # 7
        [3, 5, 6],  # 8
        [4, 6, 7]   # 9
    ]
    # Inner star: 5-7-9-6-8-5
    # 5: neighbors 7, 8 (and 0 for spoke)
    # 7: neighbors 5, 9 (and 2 for spoke)
    # 9: neighbors 7, 6 (and 4 for spoke)
    # 6: neighbors 9, 8 (and 1 for spoke)
    # 8: neighbors 6, 5 (and 3 for spoke)
    petersen_adj = [
        [1, 4, 5],  # 0
        [0, 2, 6],  # 1
        [1, 3, 7],  # 2
        [2, 4, 8],  # 3
        [0, 3, 9],  # 4
        [0, 7, 8],  # 5
        [1, 9, 8],  # 6
        [2, 5, 9],  # 7
        [3, 5, 6],  # 8
        [4, 6, 7]   # 9
    ]
    
    # K3,3 (6 vertices, 3-regular)
    k33_adj = [
        [3, 4, 5],
        [3, 4, 5],
        [3, 4, 5],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ]
    
    for base_adj, base_name in [(petersen_adj, "Petersen"), (k33_adj, "K3,3")]:
        if _time_left() < 15.0:
            break
        base_n = len(base_adj)
        # Try different lift sizes
        for m in range(2, 20):
            if _time_left() < 5.0:
                break
            n_lift = base_n * m
            if n_lift < moore:
                continue
            result = _try_lift(base_adj, m, g)
            if result is not None:
                n_lift, adj_lift = result
                if best_n is None or n_lift < best_n:
                    best_n = n_lift
                    edges = []
                    for i in range(n_lift):
                        for j in adj_lift[i]:
                            if i < j:
                                edges.append((i, j))
                    best_edges = edges
                    return n_lift, edges

    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    if best_n is not None:
        return best_n, best_edges
    
    return _luzw_construct(k, g)


def _luzw_construct(k, g):
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
