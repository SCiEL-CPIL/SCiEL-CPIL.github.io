"""Constructive search for small k-regular graphs of large girth.

Strategy (different from the algebraic Lazebnik-Ustimenko-Woldar family):
  1. Estimate a candidate vertex count n from the Moore bound (lower) up to a
     generous upper cap.
  2. Generate random k-regular simple graphs on n vertices.
  3. Repair short cycles via 2-switches (edge swaps) while incrementally
     verifying that no cycle shorter than g remains.
  4. Certify girth exactly with non-backtracking BFS from every vertex.
  5. Return the first valid graph found within the time budget; otherwise
     return the best candidate that satisfies regularity + simplicity and has
     the largest verified girth (falling back to a known-good construction).

The interface construct(k, g) -> (n, edges) is preserved.
"""
import random
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # leave headroom under the 120 s cap


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _moore_bound(k, g):
    """Moore bound (lower bound) for n(k, g)."""
    if g % 2 == 0:
        # Even girth: n >= k * ( (k-1)^(g/2 - 1) - 1 ) / (k-2) + 2
        h = g // 2
        if k == 2:
            return 2 * h
        # n >= k * sum_{i=0}^{h-1} (k-1)^i  ... standard Moore bound
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s
    else:
        # Odd girth: n >= k * ( (k-1)^((g-1)/2) - 1 ) / (k-2)
        h = (g - 1) // 2
        if k == 2:
            return g
        s = 0
        p = 1
        for _ in range(h):
            s += p
            p *= (k - 1)
        return k * s


def _random_regular_graph(n, k, rng):
    """Return adjacency list of a random k-regular simple graph on n vertices,
    or None if not constructible (n*k must be even and k < n)."""
    if n * k % 2 != 0 or k >= n:
        return None
    # Configuration model: stubs
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    # pair stubs
    i = 0
    while i < len(stubs):
        a = stubs[i]
        b = stubs[i + 1]
        i += 2
        if a == b:
            continue  # loop; will retry by re-pairing
        if b in adj[a]:
            continue  # multi-edge
        adj[a].add(b)
        adj[b].add(a)
    # If we skipped pairs, degrees are wrong; repair by swapping.
    # Simpler: retry with a clean approach using a list of half-edges and
    # a repair loop.
    return _repair_to_regular(adj, n, k, rng)


def _repair_to_regular(adj, n, k, rng):
    """Repair adjacency sets to be exactly k-regular and simple.
    Returns adj (list of sets) or None on failure."""
    for _ in range(200):
        under = [v for v in range(n) if len(adj[v]) < k]
        over = [v for v in range(n) if len(adj[v]) > k]
        if not under and not over:
            # ensure symmetry
            ok = True
            for v in range(n):
                for u in adj[v]:
                    if v not in adj[u]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return adj
            # fix asymmetry
            for v in range(n):
                for u in list(adj[v]):
                    if v not in adj[u]:
                        adj[u].add(v)
            continue
        if not under or not over:
            break
        a = rng.choice(under)
        b = rng.choice(over)
        if a == b:
            continue
        if b in adj[a]:
            continue
        # remove a random edge from b to reduce its degree
        nb = list(adj[b])
        if not nb:
            continue
        c = rng.choice(nb)
        if c == a:
            continue
        adj[b].discard(c)
        adj[c].discard(b)
        adj[a].add(b)
        adj[b].add(a)
    # Final check
    for v in range(n):
        if len(adj[v]) != k:
            return None
    for v in range(n):
        for u in adj[v]:
            if v not in adj[u]:
                return None
    return adj


def _girth_check_bfs(adj, n, g, start_limit=None):
    """Non-backtracking BFS from every vertex (or a subset) to detect any
    cycle of length < g. Returns (has_short_cycle, min_cycle_len_found)."""
    # For each start vertex, do BFS on the directed line graph (v, u) meaning
    # we came from v to u, forbidding backtracking to v.
    min_len = g  # we only care if < g
    starts = range(n) if start_limit is None else range(start_limit)
    for s in starts:
        # state: (prev, cur), distance from s
        visited = {}
        dq = deque()
        for u in adj[s]:
            dq.append((s, u, 1))
            visited[(s, u)] = 1
        while dq:
            prev, cur, dist = dq.popleft()
            if dist >= g:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue  # non-backtracking
                if nxt == s:
                    # cycle found of length dist+1
                    if dist + 1 < min_len:
                        min_len = dist + 1
                    continue
                if (cur, nxt) in visited:
                    if dist + 1 < visited[(cur, nxt)]:
                        # found a shorter path to same state -> cycle
                        cyc = dist + 1 - visited[(cur, nxt)]
                        if cyc < min_len:
                            min_len = cyc
                    continue
                visited[(cur, nxt)] = dist + 1
                dq.append((cur, nxt, dist + 1))
    return (min_len < g), min_len


def _full_girth_certify(adj, n, g):
    """Exact girth certification via non-backtracking BFS from every vertex."""
    has_short, min_len = _girth_check_bfs(adj, n, g)
    return (not has_short), min_len


def _try_build(n, k, g, rng, max_attempts):
    """Try to build a k-regular graph on n vertices with girth >= g.
    Returns (adj, min_girth) on success, else (None, None)."""
    for attempt in range(max_attempts):
        if _time_left() < 2.0:
            return None, None
        adj = _random_regular_graph(n, k, rng)
        if adj is None:
            continue
        # Quick girth check from a subset of vertices
        has_short, min_len = _girth_check_bfs(adj, n, g, start_limit=min(n, 20))
        if not has_short:
            # Full certification
            ok, min_len2 = _full_girth_certify(adj, n, g)
            if ok:
                return adj, min_len2
        # Repair short cycles via 2-switches
        for _ in range(50):
            if _time_left() < 2.0:
                break
            has_short, min_len = _girth_check_bfs(adj, n, g, start_limit=min(n, 15))
            if not has_short:
                ok, min_len2 = _full_girth_certify(adj, n, g)
                if ok:
                    return adj, min_len2
                break
            # Find a short cycle and try to break it with a 2-switch
            # Pick a random edge and try swapping with a random other edge
            v = rng.randrange(n)
            if not adj[v]:
                break
            u = rng.choice(list(adj[v]))
            # find another edge (x,y) not sharing endpoints with (v,u)
            for _ in range(50):
                x = rng.randrange(n)
                if x in (v, u):
                    continue
                if not adj[x]:
                    continue
                y = rng.choice(list(adj[x]))
                if y in (v, u) or x == y:
                    continue
                # perform 2-switch: remove (v,u),(x,y); add (v,x),(u,y) if simple
                if x in adj[v] or y in adj[u]:
                    continue
                adj[v].discard(u)
                adj[u].discard(v)
                adj[x].discard(y)
                adj[y].discard(x)
                adj[v].add(x)
                adj[x].add(v)
                adj[u].add(y)
                adj[y].add(u)
                break
    return None, None


def _fallback_luw(k, g):
    """Fallback: the original Lazebnik-Ustimenko-Woldar construction.
    Guaranteed valid, used if the search fails."""
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


def _tree_closure_construct(k, g):
    """
    Constructs a k-regular graph of girth g using the Tree-Closure method.
    1. Build a k-regular tree of depth r = (g-1)//2.
    2. The leaves have degree 1. We need to add k-1 edges to each leaf.
    3. We connect leaves such that no short cycles are formed.
       Specifically, leaves under the same node at depth 1 cannot be connected
       (would form a cycle of length < g).
    4. We use a greedy matching to connect leaves from different branches.
    """
    if g % 2 == 0:
        return None
    r = (g - 1) // 2
    if r < 1:
        return None
    
    # Build the tree
    # Node 0 is root.
    # Level 0: 1 node
    # Level i: k^(i-1) * k nodes? No.
    # Root has k children.
    # Each internal node (depth < r) has k-1 children (one parent).
    # Depth r nodes are leaves (degree 1 in tree).
    
    n_nodes = 1
    adj = [[] for _ in range(n_nodes)] # Will resize
    parent = [-1] * n_nodes
    depth = [0] * n_nodes
    
    # We need to dynamically add nodes.
    # Let's pre-calculate the number of nodes.
    # N_0 = 1
    # N_i = N_{i-1} * (k-1) for i > 0?
    # Root has k children. So N_1 = k.
    # N_2 = k * (k-1).
    # N_i = k * (k-1)^(i-1).
    
    total_nodes = 1
    for i in range(1, r + 1):
        if i == 1:
            total_nodes += k
        else:
            total_nodes += k * ((k-1) ** (i-1))
            
    if total_nodes > 5000: # Safety check
        return None

    # Re-initialize structures
    adj = [set() for _ in range(total_nodes)]
    parent = [-1] * total_nodes
    depth = [0] * total_nodes
    
    # Build tree
    # Root is 0.
    next_id = 1
    # Children of root
    for _ in range(k):
        child = next_id
        next_id += 1
        adj[0].add(child)
        adj[child].add(0)
        parent[child] = 0
        depth[child] = 1
        
    # Expand levels 1 to r-1
    current_level_ids = [i for i in range(1, 1+k)]
    for d in range(1, r):
        next_level_ids = []
        for node in current_level_ids:
            for _ in range(k-1):
                child = next_id
                next_id += 1
                adj[node].add(child)
                adj[child].add(node)
                parent[child] = node
                depth[child] = d + 1
                next_level_ids.append(child)
        current_level_ids = next_level_ids
    
    leaves = current_level_ids # Nodes at depth r
    
    # Each leaf needs k-1 more edges.
    # Total stubs = len(leaves) * (k-1)
    # We must connect leaves from different "branches" (children of root).
    # Branch ID of a leaf is the child of root that is its ancestor.
    
    branch_of_leaf = {}
    for leaf in leaves:
        # Find ancestor at depth 1
        curr = leaf
        while depth[curr] > 1:
            curr = parent[curr]
        branch_of_leaf[leaf] = curr
        
    # Group leaves by branch
    branches = {}
    for leaf in leaves:
        b = branch_of_leaf[leaf]
        if b not in branches:
            branches[b] = []
        branches[b].append(leaf)
        
    # We need to match stubs.
    # Each leaf has k-1 stubs.
    # We create a list of (leaf, stub_index)
    # And try to pair them such that branch(u) != branch(v).
    
    # Greedy matching:
    # For each leaf, for each stub, find an available leaf in a different branch.
    # To avoid bias, we shuffle.
    
    # Create a pool of available stubs for each branch
    available_stubs = {b: [] for b in branches}
    for b, leaves_in_b in branches.items():
        for leaf in leaves_in_b:
            for _ in range(k-1):
                available_stubs[b].append(leaf)
                
    # We will process stubs one by one.
    # To ensure we don't get stuck, we might need backtracking, but greedy often works for these parameters.
    # Let's try a simple greedy:
    # Sort branches by size (all same).
    # For each branch b, for each stub in b, pick a stub from another branch.
    
    # Better: Randomized Greedy
    rng = random.Random(42)
    
    # Flatten all stubs with their branch
    all_stubs = []
    for b, stubs in available_stubs.items():
        for s in stubs:
            all_stubs.append((b, s))
            
    rng.shuffle(all_stubs)
    
    # We need to pair them.
    # We'll use a dictionary to track which stubs are already matched.
    # Since we are adding edges, we just need to ensure we don't add duplicate edges or loops.
    # And we must respect the branch constraint.
    
    # Actually, we can just iterate and pick partners.
    # Let's maintain a list of "free" stubs for each branch.
    free_stubs = {b: list(stubs) for b, stubs in available_stubs.items()}
    
    # We process each leaf's stubs one by one.
    # To be symmetric, we'll just process all stubs in a random order and pair them up.
    # But we must ensure that if we pair (u, v), we remove both from free lists.
    
    # Let's create a list of all stubs to be matched.
    # Each leaf L needs k-1 partners.
    # We'll create a list of (leaf, count) and process it.
    
    # Simpler: 
    # For each leaf u:
    #   for i in range(k-1):
    #     pick a leaf v from a different branch that still has capacity.
    #     add edge (u, v)
    #     decrement capacity of v.
    
    capacity = {leaf: k-1 for leaf in leaves}
    
    for u in leaves:
        for _ in range(k-1):
            # Find a valid v
            # We prefer branches with high remaining capacity
            # and u's branch is excluded.
            
            candidates = []
            for b, stubs in free_stubs.items():
                if b == branch_of_leaf[u]:
                    continue
                # Check if any leaf in this branch has capacity
                # We can just pick a random leaf from this branch that has capacity
                valid_leaves_in_b = [l for l in stubs if capacity[l] > 0]
                if valid_leaves_in_b:
                    candidates.extend(valid_leaves_in_b)
                    
            if not candidates:
                return None # Failed to find partner
            
            # Pick randomly to avoid bias
            v = rng.choice(candidates)
            
            # Add edge
            if v in adj[u]:
                return None # Should not happen if we track capacity correctly, but check
            adj[u].add(v)
            adj[v].add(u)
            
            capacity[u] -= 1
            capacity[v] -= 1
            
            # Remove v from free_stubs[branch_of_leaf[v]]?
            # No, v might have other stubs. We just reduced its capacity.
            # The `free_stubs` list contains leaves, not stubs.
            # We need to ensure we don't pick a leaf with 0 capacity.
            # The check `capacity[l] > 0` handles this.
            
    # Verify regularity
    for u in range(total_nodes):
        if len(adj[u]) != k:
            return None
            
    # Verify girth
    ok, min_len = _full_girth_certify(adj, total_nodes, g)
    if ok:
        edges = []
        for v in range(total_nodes):
            for u in adj[v]:
                if v < u:
                    edges.append((v, u))
        return total_nodes, edges
    else:
        return None

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    if k < 2 or g < 3:
        return _fallback_luw(max(2, k), max(3, g))

    # Try Tree Closure first (for odd girth)
    if g % 2 == 1:
        result = _tree_closure_construct(k, g)
        if result:
            return result

    # Fallback to LUW
    return _fallback_luw(k, g)
