"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math as _math
from collections import deque, defaultdict

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _moore_bound(k, g):
    if g % 2 == 0:
        return 2 * (k * (k-1)**(g//2 - 1))
    else:
        return k + 2 * (k-1)**(g//2)

def _build_moore_tree(k, g):
    """Build the k-regular Moore tree of depth g//2 rooted at 0.
    Returns (n, adj) where adj is a list of lists.
    Root has k neighbours; internal non-root has k-1 children; leaves have 1 parent."""
    depth = g // 2
    adj = []
    def add_node():
        adj.append([])
        return len(adj) - 1
    root = add_node()
    frontier = [root]
    for d in range(depth):
        next_frontier = []
        for v in frontier:
            if v == root:
                children = k
            else:
                children = k - 1
            for _ in range(children):
                c = add_node()
                adj[v].append(c)
                adj[c].append(v)
                next_frontier.append(c)
        frontier = next_frontier
    n = len(adj)
    leaves = list(frontier)
    return n, adj, leaves, root

def _short_cycle_exists(adj, n, a, b, g):
    """Check if adding edge (a,b) creates a cycle of length < g.
    We do BFS from a with prev=-1, tracking depth; if we reach b at depth d
    and d+1 < g, there's a short cycle. Also we must not go through b before
    the final step (since b is not yet connected to a)."""
    # BFS from a, not using edge (a,b)
    dist = {a: 0}
    dq = deque([a])
    while dq:
        v = dq.popleft()
        d = dist[v]
        if d >= g - 1:
            continue
        for w in adj[v]:
            if w == b and d + 1 < g:
                return True
            if w in dist:
                continue
            dist[w] = d + 1
            dq.append(w)
    return False

def _greedy_tree_extension(n, adj, leaves, k, g, rng, deadline):
    """Try to extend the Moore tree to a k-regular graph by pairing leaves.
    Each leaf needs (k-1) more edges (since it has 1 parent).
    Total stubs = len(leaves) * (k-1). Pair them randomly, skipping if
    short cycle or self-loop or duplicate edge would result.
    Returns (n, adj, edges) or None if stuck."""
    # Count stubs per leaf
    stubs_per_leaf = k - 1
    leaf_ids = list(leaves)
    rng.shuffle(leaf_ids)
    # Build a list of stubs: (leaf, stub_index)
    stubs = []
    for leaf in leaf_ids:
        for s in range(stubs_per_leaf):
            stubs.append(leaf)
    rng.shuffle(stubs)
    
    # Track existing edges for quick duplicate check
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    # Track remaining stubs per leaf
    remaining = {leaf: stubs_per_leaf for leaf in leaf_ids}
    used = [False] * len(stubs)
    
    # Process stubs in order, greedily pairing
    for i in range(len(stubs)):
        if used[i]:
            continue
        a = stubs[i]
        if remaining[a] == 0:
            used[i] = True
            continue
        # Find a partner stub j > i
        found = False
        for j in range(i + 1, len(stubs)):
            if used[j]:
                continue
            b = stubs[j]
            if remaining[b] == 0:
                used[j] = True
                continue
            if a == b:
                used[j] = True
                continue
            if (a, b) in edge_set or (b, a) in edge_set:
                used[j] = True
                continue
            if _short_cycle_exists(adj, n, a, b, g):
                used[j] = True
                continue
            # Add edge
            adj[a].append(b)
            adj[b].append(a)
            edge_set.add((a, b))
            remaining[a] -= 1
            remaining[b] -= 1
            used[i] = True
            used[j] = True
            found = True
            break
        if not found:
            return None
        if _time.time() > deadline:
            return None
    # Verify all leaves have degree k
    for leaf in leaf_ids:
        if len(adj[leaf]) != k:
            return None
    # Verify all vertices have degree k
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graph of girth >= g."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 145
    
    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges
    
    moore = _moore_bound(k, g)
    # n must be even for 3-regular
    if moore % 2 != 0:
        moore += 1
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: Randomized greedy tree-extension from the Moore tree
    # The Moore tree has exactly the Moore bound number of vertices.
    # We try multiple random pairings of leaves.
    n_tree, adj_tree, leaves, root = _build_moore_tree(k, g)
    if n_tree >= moore and n_tree < best_n:
        for attempt in range(200):
            if _time.time() > deadline - 15:
                break
            rng = _random.Random(attempt * 7919 + 13)
            # Deep copy adj_tree
            adj_try = [list(neigh) for neigh in adj_tree]
            result = _greedy_tree_extension(n_tree, adj_try, leaves, k, g, rng, deadline - 15)
            if result is not None:
                # Verify girth
                if _verify_girth_fast(adj_try, n_tree, g):
                    edges = []
                    for u in range(n_tree):
                        for v in adj_try[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n_tree
                    best_result = (n_tree, edges)
                    break
    
    # Strategy 2: Try larger n with random 3-regular graphs
    if best_result is None or best_n > moore + 200:
        for n_try in range(moore, moore + 400, 2):
            if _time.time() > deadline - 15:
                break
            if n_try >= best_n:
                break
            for attempt in range(50):
                if _time.time() > deadline - 15:
                    break
                rng = _random.Random(attempt * 104729 + n_try * 17)
                stubs = []
                for v in range(n_try):
                    for _ in range(3):
                        stubs.append(v)
                rng.shuffle(stubs)
                adj = [set() for _ in range(n_try)]
                edges = []
                valid = True
                for i in range(0, n_try * 3, 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                if not valid:
                    continue
                # Check connectivity
                visited = set([0])
                stack = [0]
                while stack:
                    v = stack.pop()
                    for w in adj[v]:
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
                if len(visited) != n_try:
                    continue
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n_try, g):
                    best_n = n_try
                    best_result = (n_try, edges)
                    break
    
    # Strategy 3: LUW fallback
    if _time.time() < deadline - 5:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
