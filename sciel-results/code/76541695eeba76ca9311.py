"""
Cage graph construction: cycle-avoiding greedy completion with 2-switch local search.
Strategy: Start from a random 3-regular graph near the Moore bound, then iteratively
find shortest cycles and break them via randomized 2-switches, preserving 3-regularity.
"""
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj_list, n, g):
    """Check girth >= g using non-backtracking BFS from each vertex."""
    for start in range(n):
        # dist[(v, prev)] = length of non-backtracking walk from start
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _find_shortest_cycle(adj_list, n):
    """Find a shortest cycle in the graph. Returns (length, list_of_vertices) or (n+1, [])."""
    best_len = n + 1
    best_cycle = []
    for start in range(n):
        # BFS to find shortest cycle through start
        # dist[v] = shortest distance from start
        # parent[v] = predecessor in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        dq = deque([start])
        found = False
        while dq and not found:
            v = dq.popleft()
            for w in adj_list[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    dq.append(w)
                elif w != parent[v] and v != parent[w]:
                    # Found a cycle
                    cycle_len = dist[v] + dist[w] + 1
                    if cycle_len < best_len:
                        best_len = cycle_len
                        # Reconstruct cycle
                        cycle = []
                        # Path from w up to LCA
                        path_w = []
                        x = w
                        while x != -1:
                            path_w.append(x)
                            x = parent[x]
                        path_v = []
                        x = v
                        while x != -1:
                            path_v.append(x)
                            x = parent[x]
                        # Find LCA
                        set_v = set(path_v)
                        lca = None
                        for node in path_w:
                            if node in set_v:
                                lca = node
                                break
                        if lca is not None:
                            # Cycle: v -> ... -> lca -> ... -> w -> v
                            cycle = []
                            x = v
                            while x != lca:
                                cycle.append(x)
                                x = parent[x]
                            cycle.append(lca)
                            x = w
                            while x != lca:
                                cycle.append(x)
                                x = parent[x]
                            best_cycle = cycle
                        found = True
                        break
    return best_len, best_cycle

def _random_3reg(n, seed):
    """Generate a random 3-regular graph on n vertices using configuration model + 2-switches."""
    rng = _random.Random(seed)
    # Configuration model
    half_edges = []
    for v in range(n):
        for _ in range(3):
            half_edges.append(v)
    rng.shuffle(half_edges)
    # Pair up
    adj = [set() for _ in range(n)]
    valid = False
    for attempt in range(200):
        rng.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n * 3, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if ok:
            # Check all degrees
            valid = all(len(adj[i]) == 3 for i in range(n))
            if valid:
                break
    if not valid:
        # Fallback: use a simpler construction
        # Ring + chords
        adj = [set() for _ in range(n)]
        for i in range(n):
            adj[i].add((i - 1) % n)
            adj[i].add((i + 1) % n)
        # Add random chords
        remaining = [i for i in range(n)]
        rng.shuffle(remaining)
        for i in range(0, n, 2):
            u, v = remaining[i], remaining[i + 1]
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
            else:
                # Try another pair
                for j in range(i + 2, n, 2):
                    u2, v2 = remaining[i], remaining[j]
                    if v2 not in adj[u2]:
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        break
                # Fix degrees
                for k in range(n):
                    while len(adj[k]) > 3:
                        # Remove an edge
                        for nb in list(adj[k]):
                            if nb in adj[k] and k in adj[nb]:
                                adj[k].discard(nb)
                                adj[nb].discard(k)
                                break
                    while len(adj[k]) < 3:
                        # Add a random edge
                        for nb in range(n):
                            if nb != k and nb not in adj[k]:
                                adj[k].add(nb)
                                adj[nb].add(k)
                                break
        valid = all(len(adj[i]) == 3 for i in range(n))
    adj_list = [sorted(adj[i]) for i in range(n)]
    return adj_list

def _two_switch(adj_list, n, edge1, edge2, rng):
    """
    Perform a 2-switch: replace edges (a,b) and (c,d) with (a,c) and (b,d)
    (or (a,d) and (b,c)), preserving 3-regularity.
    Returns True if switch was made, False otherwise.
    """
    a, b = edge1
    c, d = edge2
    # Check no loops or multi-edges
    if a == c or a == d or b == c or b == d:
        return False
    # Option 1: (a,c) and (b,d)
    if c not in adj_list[a] and d not in adj_list[b]:
        adj_list[a].remove(b)
        adj_list[b].remove(a)
        adj_list[a].append(c)
        adj_list[c].append(a)
        adj_list[b].append(d)
        adj_list[d].append(b)
        for v in (a, b, c, d):
            adj_list[v].sort()
        return True
    # Option 2: (a,d) and (b,c)
    if d not in adj_list[a] and c not in adj_list[b]:
        adj_list[a].remove(b)
        adj_list[b].remove(a)
        adj_list[a].append(d)
        adj_list[d].append(a)
        adj_list[b].append(c)
        adj_list[c].append(b)
        for v in (a, b, c, d):
            adj_list[v].sort()
        return True
    return False

def _break_cycle_2switch(adj_list, n, cycle, rng):
    """
    Try to break a cycle using 2-switches. Pick two edges on the cycle and
    try to swap them with edges off the cycle.
    Returns True if a switch was made.
    """
    if len(cycle) < 4:
        return False
    # Get edges on the cycle
    cycle_edges = []
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        if v in adj_list[u]:
            cycle_edges.append((u, v))
    
    if len(cycle_edges) < 2:
        return False
    
    # Try random pairs of cycle edges
    rng.shuffle(cycle_edges)
    for i in range(min(len(cycle_edges), 10)):
        e1 = cycle_edges[i]
        # Find a non-cycle edge incident to one of e1's endpoints
        a, b = e1
        for v in (a, b):
            for nb in adj_list[v]:
                if nb == a and v == b or nb == b and v == a:
                    continue
                # Edge (v, nb) is not on the cycle (or is it?)
                # Check if (v, nb) is a cycle edge
                is_cycle_edge = False
                for ce in cycle_edges:
                    if (ce[0] == v and ce[1] == nb) or (ce[0] == nb and ce[1] == v):
                        is_cycle_edge = True
                        break
                if is_cycle_edge:
                    continue
                # Try 2-switch between e1 and (v, nb)
                if _two_switch(adj_list, n, e1, (v, nb), rng):
                    return True
    return False

def _local_search(adj_list, n, g, deadline, max_iter=50000):
    """
    Local search: iteratively find shortest cycle and try to break it with 2-switches.
    """
    rng = _random.Random(12345)
    best_girth = n + 1
    best_adj = [list(x) for x in adj_list]
    
    for iteration in range(max_iter):
        if _time.time() > deadline - 5:
            break
        
        # Find shortest cycle
        cycle_len, cycle = _find_shortest_cycle(adj_list, n)
        
        if cycle_len >= g:
            # Success!
            return [list(x) for x in adj_list], cycle_len
        
        if cycle_len < best_girth:
            best_girth = cycle_len
            best_adj = [list(x) for x in adj_list]
        
        # Try to break the cycle
        if not _break_cycle_2switch(adj_list, n, cycle, rng):
            # Try random 2-switches
            made_switch = False
            for _ in range(20):
                # Pick two random edges
                e1 = (rng.randint(0, n-1), rng.randint(0, n-1))
                e2 = (rng.randint(0, n-1), rng.randint(0, n-1))
                # Make them actual edges
                if e1[1] in adj_list[e1[0]] and e2[1] in adj_list[e2[0]]:
                    if _two_switch(adj_list, n, e1, e2, rng):
                        made_switch = True
                        break
            if not made_switch:
                # Try more structured random switches
                for _ in range(50):
                    u = rng.randint(0, n-1)
                    nb1 = rng.choice(adj_list[u])
                    v = rng.randint(0, n-1)
                    nb2 = rng.choice(adj_list[v])
                    if u != v and nb1 != v and nb2 != u:
                        if _two_switch(adj_list, n, (u, nb1), (v, nb2), rng):
                            made_switch = True
                            break
    
    return best_adj, best_girth

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
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

def _adj_to_edges(adj_list, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                edges.append((u, v))
    return edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 145  # Safety margin
    
    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # LUW construction as a strong baseline
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Try cycle-avoiding greedy completion with local search
    # Start from various sizes near the Moore bound
    for start_n in range(max(moore, 100), min(n_luw, moore + 400), 16):
        if _time.time() > deadline - 30:
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if _time.time() > deadline - 25:
                break
            
            adj_list = _random_3reg(start_n, seed)
            
            # Check if already valid
            if _verify_girth(adj_list, start_n, g):
                if start_n < best_n:
                    best_n = start_n
                    edges = _adj_to_edges(adj_list, start_n)
                    best_result = (best_n, edges)
                break
            
            # Local search to improve girth
            adj_improved, girth_found = _local_search(adj_list, start_n, g, deadline - 10, max_iter=30000)
            
            if girth_found >= g:
                if start_n < best_n:
                    best_n = start_n
                    edges = _adj_to_edges(adj_improved, start_n)
                    best_result = (best_n, edges)
                break
    
    return best_result
