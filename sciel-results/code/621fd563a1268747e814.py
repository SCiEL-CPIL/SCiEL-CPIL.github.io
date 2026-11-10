"""Stronger program: LUW baseline + optimized girth check + targeted search for smaller k-regular girth-g graphs."""
import time
import random
from collections import deque, defaultdict

_DEADLINE = 115  # safety margin before 120s wall-clock kill

def _time_left():
    return _DEADLINE - time.time()

def _moore_lower(k, g):
    """Moore bound: minimum possible n for k-regular graph with girth g."""
    if g % 2 == 0:
        r = g // 2
        # n >= k * (k-1)^(r-1) / 2 * ... simplified: for even girth g=2r
        # n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 / k ... use standard form
        # For girth g (even): n >= k * sum_{i=0}^{r-1} (k-1)^i = k * ((k-1)^r - 1)/(k-2)
        # Actually the Moore bound for k-regular, girth g:
        # if g even (g=2r): n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 ... let me use the tree bound
        # A tree of depth r-1 from each vertex: number of distinct vertices >= 2 * sum_{i=0}^{r-1}(k-1)^i
        # = 2 * ((k-1)^r - 1)/(k-2) for k>2
        r = g // 2
        if k <= 2:
            return g
        return 2 * ((k - 1) ** r - 1) // (k - 2)
    else:
        # g odd (g=2r+1): center vertex, tree of depth r
        # n >= 1 + k * sum_{i=0}^{r-1}(k-1)^i = 1 + k * ((k-1)^r - 1)/(k-2)
        r = (g - 1) // 2
        if k <= 2:
            return g
        return 1 + k * ((k - 1) ** r - 1) // (k - 2)

def _check_girth(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth >= g. Optimized with bit-packed visited states."""
    if n == 0:
        return True
    max_depth = (g - 1) // 2
    # Use a 2D array for visited: visited[depth][node] tracks if (node, prev) was seen
    # Actually, we need to track (node, prev) pairs. Use a set of encoded integers.
    # Encode (node, prev) as node * n + prev (with prev=-1 mapped to n)
    for start in range(n):
        # BFS state: (node, prev_node) to avoid backtracking
        # Encode state as node * n + (prev + 1) where prev=-1 becomes 0, prev>=0 becomes prev+1
        # But simpler: use a set of tuples, but check early exit
        visited = set()
        queue = deque()
        queue.append((start, -1))
        visited.add((start, -1))
        depth = 0
        while queue and depth <= max_depth:
            next_queue = deque()
            for _ in range(len(queue)):
                node, prev = queue.popleft()
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle of length depth+1 (or less)
                        if depth + 1 < g:
                            return False
                        continue
                    visited.add((nb, node))
                    next_queue.append((nb, node))
            queue = next_queue
            depth += 1
    return True

def _check_degree(adj, n, k):
    """Check every vertex has degree exactly k."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _check_simple(adj, n):
    """Check no loops, no repeated edges."""
    for i in range(n):
        if i in adj[i]:
            return False
        s = set(adj[i])
        if len(s) != len(adj[i]):
            return False
    return True

def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if not _check_simple(adj, n):
        return False
    if not _check_degree(adj, n, k):
        return False
    if not _check_girth(adj, n, g):
        return False
    return True

def _lazebnik_construct(k, g):
    """LUW graph construction - fast, always valid for prime k or k=4."""
    q = k
    # d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
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

def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g.
    Tries to build graphs from the Moore bound upward using configuration model with girth-aware pairing."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    # Try a range of n values, starting from lower bound
    # Focus on n values that are multiples of k (needed for k-regular)
    for offset in range(0, 300):
        if time.time() - start_time > time_budget:
            break
        target_n = lower + offset
        if target_n < k + 1:
            continue
        if k * target_n % 2 != 0:
            continue
        if target_n > 500:
            break
        
        # Try multiple random initializations with girth-aware configuration
        for attempt in range(5):
            if time.time() - start_time > time_budget:
                break
            
            # Build a random k-regular graph on target_n vertices with girth awareness
            adj = _random_k_regular_girth_aware(target_n, k, g)
            if adj is None:
                continue
            
            # Verify girth
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break  # Found valid for this target_n
    
    if best_n is not None:
        return best_n, best_edges
    return None, None

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model + simplification."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Randomly pair stubs
    for _ in range(100):  # try 100 times to get a simple graph
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Convert to lists
            return [list(s) for s in adj]
    return None

def _random_k_regular_girth_aware(n, k, g, max_attempts=50):
    """Generate a random k-regular graph with girth awareness using iterative pairing."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    # For girth >= g, we need to avoid creating short cycles
    # Use a smarter configuration model that checks for short cycles during construction
    max_depth = (g - 1) // 2
    
    for attempt in range(max_attempts):
        # Build adjacency incrementally
        adj = [set() for _ in range(n)]
        # Track stubs as list of (vertex, stub_id)
        stubs = [(i, j) for i in range(n) for j in range(k)]
        random.shuffle(stubs)
        
        valid = True
        for idx in range(0, len(stubs), 2):
            u, _ = stubs[idx]
            v, _ = stubs[idx + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            # Check if adding edge (u,v) creates a short cycle
            # Quick check: if there's a path from u to v of length < g-1 not using this edge
            # For efficiency, only check for small g
            if g <= 12:
                if _would_create_short_cycle(adj, n, u, v, g):
                    valid = False
                    break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            # Verify full girth
            if _check_girth([list(s) for s in adj], n, g):
                return [list(s) for s in adj]
    return None

def _would_create_short_cycle(adj, n, u, v, g):
    """Check if adding edge (u,v) to current adj creates a cycle of length < g."""
    # BFS from u to v, avoiding the edge (u,v) itself
    # If we find a path of length < g-1, then adding (u,v) creates a cycle of length <= g-1 < g
    max_path_len = g - 2  # path length + 1 for the new edge < g
    if max_path_len < 1:
        return False
    visited = {u: 0}
    queue = deque([(u, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_path_len:
            continue
        for nb in adj[node]:
            if nb == v and dist + 1 <= max_path_len:
                # Found path from u to v of length dist+1
                # Cycle length = dist + 1 + 1 = dist + 2
                if dist + 2 < g:
                    return True
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))
    return False

def _repair_girth(adj, n, k, g, elapsed, budget):
    """Try to repair a k-regular graph to have girth >= g by swapping edges.
    Uses 2-switch operations to eliminate short cycles."""
    if elapsed > budget:
        return None
    remaining = budget - elapsed
    max_iters = min(2000, int(remaining * 20))
    
    adj = [list(s) for s in adj]  # copy
    # Convert to sets for faster lookup
    adj_sets = [set(s) for s in adj]
    
    for _ in range(max_iters):
        if time.time() - (budget - remaining) > budget:
            break
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return [list(s) for s in adj_sets]
        # Try to fix by 2-switch: remove edges (u,v) and (w,x), add (u,w) and (v,x)
        # Pick two edges from the cycle that are non-adjacent
        if len(cycle) < 4:
            # Cycle too short, try random 2-switch
            break
        # Pick random pair of non-adjacent edges in cycle
        edge_pairs = []
        for i in range(len(cycle)):
            for j in range(i + 2, len(cycle)):
                if j - i == 1 or (i == 0 and j == len(cycle) - 1):
                    continue
                edge_pairs.append((i, j))
        if not edge_pairs:
            break
        i, j = random.choice(edge_pairs)
        u1, v1 = cycle[i], cycle[(i + 1) % len(cycle)]
        u2, v2 = cycle[j], cycle[(j + 1) % len(cycle)]
        # Remove edges (u1,v1) and (u2,v2)
        adj_sets[u1].discard(v1)
        adj_sets[v1].discard(u1)
        adj_sets[u2].discard(v2)
        adj_sets[v2].discard(u2)
        # Try adding (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
        found = False
        for pairing in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
            (a, b), (c, d) = pairing
            if a == b or c == d:
                continue
            if b in adj_sets[a] or d in adj_sets[c]:
                continue
            # Tentatively add
            adj_sets[a].add(b)
            adj_sets[c].add(d)
            adj_sets[b].add(a)
            adj_sets[d].add(c)
            # Check degrees are still k
            if all(len(adj_sets[x]) == k for x in range(n)):
                # Check girth
                adj_list = [list(s) for s in adj_sets]
                if _check_girth(adj_list, n, g):
                    found = True
                    break
            # Revert
            adj_sets[a].discard(b)
            adj_sets[c].discard(d)
            adj_sets[b].discard(a)
            adj_sets[d].discard(c)
        if not found:
            # Revert removals
            adj_sets[u1].add(v1)
            adj_sets[v1].add(u1)
            adj_sets[u2].add(v2)
            adj_sets[v2].add(u2)
    
    if _check_girth([list(s) for s in adj_sets], n, g):
        return [list(s) for s in adj_sets]
    return None

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g if one exists. Returns list of vertices or None."""
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS with parent tracking
        parent = {start: None}
        depth = {start: 0}
        queue = deque([(start, -1)])
        while queue:
            node, prev = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb in parent:
                    # Found a cycle
                    cycle_len = depth[node] + depth[nb] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cycle.append(cur)
                        cur = nb
                        while cur != node:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
                else:
                    parent[nb] = node
                    depth[nb] = depth[node] + 1
                    queue.append((nb, node))
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing n."""
    start_time = time.time()
    
    # Always get LUW as baseline
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try greedy search with remaining time
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 8  # 8s safety margin
    if remaining > 5:
        greedy_n, greedy_edges = _greedy_search(k, g, time_budget=remaining)
        if greedy_n is not None and greedy_n < best_n:
            best_n, best_edges = greedy_n, greedy_edges
    
    return best_n, best_edges
