"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _moore_bound(k, g):
    """Calculate the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth: g = 2r
        r = g // 2
        # n >= 1 + k * ( (k-1)^{r-1} - 1 ) / (k-2)  -- wait, formula is:
        # n >= 2 * ( (k-1)^r - 1 ) / (k-2)
        if k == 2:
            return g
        return 2 * ((k - 1)**r - 1) // (k - 2)
    else:
        # Odd girth: g = 2r + 1
        r = (g - 1) // 2
        # n >= 1 + k * ( (k-1)^r - 1 ) / (k-2)
        if k == 2:
            return g
        return 1 + k * ((k - 1)**r - 1) // (k - 2)

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth(n, adj, g):
    """
    Returns True if girth >= g, False otherwise.
    Uses BFS from each vertex to find shortest cycle.
    """
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_random_regular_graph(n, k, seed=None):
    """
    Generates a random k-regular simple graph on n vertices using Configuration Model.
    Returns (edges, adj) if successful, else None.
    """
    if seed is not None:
        random.seed(seed)
    
    # Total stubs must be even
    if (n * k) % 2 != 0:
        return None
        
    # Create stubs: list of (vertex, stub_id)
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    # Shuffle
    random.shuffle(stubs)
    
    # Pair them up
    edges = []
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        edges.append((u, v))
        
    # Check for multi-edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
        
    adj = _build_adjacency(n, edges)
    
    # Verify degree (should be k by construction, but good to be safe)
    for v in range(n):
        if len(adj[v]) != k:
            return None
            
    return edges, adj

def _repair_girth(n, k, edges, adj, g, timeout=5.0):
    """
    Attempts to repair the graph so that girth >= g.
    Uses local rewiring (2-switches) to break short cycles.
    Returns (edges, adj) if successful, else None.
    """
    start_time = time.time()
    
    # We need to work with a mutable structure.
    # adj is a list of lists.
    # We will track edges as a set of frozensets or tuples for O(1) lookup.
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    def has_short_cycle():
        if time.time() - start_time > timeout:
            return True # Assume failed if timeout
        return not _check_girth(n, adj, g)
        
    if not has_short_cycle():
        return edges, adj
        
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            return None
            
        if not has_short_cycle():
            return edges, adj
            
        # Find a short cycle to break
        # We can just pick a random edge and see if it's part of a short cycle?
        # Or better, find the shortest cycle explicitly.
        # For efficiency in repair, we can just pick a random edge (u,v) 
        # and try to rewire it with a random non-adjacent edge (x,y).
        
        # To be effective, we should target edges in short cycles.
        # Let's find one short cycle.
        short_cycle = None
        for s in range(n):
            if time.time() - start_time > timeout:
                return None
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            found = False
            while q and not found:
                u = q.popleft()
                du = dist[u]
                if du >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            path1 = []
                            cur = u
                            while cur != s:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.append(s)
                            
                            path2 = []
                            cur = v
                            while cur != s:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.append(s)
                            
                            # Cycle is path1 + path2 reversed (excluding s)
                            # Actually, the cycle is u...s...v...u
                            # Edges in cycle:
                            # (parent[x], x) for x in path1
                            # (parent[y], y) for y in path2
                            # And (u, v) is the back edge? No, (u,v) is the non-tree edge.
                            # The cycle edges are:
                            # The path from s to u, the path from s to v, and the edge (u,v).
                            # Wait, the cycle is u -> ... -> s -> ... -> v -> u.
                            # Edges:
                            # 1. (u, v)
                            # 2. Path s to u: (s, p1), ..., (pu, u)
                            # 3. Path s to v: (s, p2), ..., (pv, v)
                            
                            cycle_edges = []
                            # Add (u,v)
                            cycle_edges.append((min(u,v), max(u,v)))
                            
                            # Add path s->u
                            cur = u
                            while cur != s:
                                p = parent[cur]
                                cycle_edges.append((min(p,cur), max(p,cur)))
                                cur = p
                                
                            # Add path s->v
                            cur = v
                            while cur != s:
                                p = parent[cur]
                                cycle_edges.append((min(p,cur), max(p,cur)))
                                cur = p
                                
                            short_cycle = cycle_edges
                            found = True
                            break
            if short_cycle:
                break
                
        if not short_cycle:
            # Should not happen if has_short_cycle returned True, 
            # but maybe it was a timeout artifact or logic error.
            # Try a random rewire anyway.
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            x = random.randint(0, n-1)
            y = random.choice(adj[x])
            if u == x or u == y or v == x or v == y:
                continue
            if (min(u,x), max(u,x)) in edge_set or (min(v,y), max(v,y)) in edge_set:
                continue
            # Perform swap
            # Remove (u,v) and (x,y)
            edge_set.remove((min(u,v), max(u,v)))
            edge_set.remove((min(x,y), max(x,y)))
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add (u,x) and (v,y)
            new_e1 = (min(u,x), max(u,x))
            new_e2 = (min(v,y), max(v,y))
            if new_e1 in edge_set or new_e2 in edge_set or u==x or v==y:
                # Revert
                edge_set.add((min(u,v), max(u,v)))
                edge_set.add((min(x,y), max(x,y)))
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                continue
                
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Check if this broke the cycle and didn't create new short ones
            # We rely on the next iteration's has_short_cycle check.
            # But to be faster, we could check locally.
            # For now, just loop.
            continue
            
        # We found a short_cycle. Pick a random edge from it to break.
        # Actually, breaking any edge in the cycle breaks the cycle.
        # Let's pick a random edge (u,v) from short_cycle.
        u, v = random.choice(short_cycle)
        
        # We need to rewire (u,v) with another edge (x,y) not in the cycle and not adjacent to u or v.
        # Find candidates for (x,y).
        # x is a neighbor of u (other than v), y is a neighbor of x? No.
        # A 2-switch swaps (u,v) and (x,y) to (u,x) and (v,y) OR (u,y) and (v,x).
        # We need x,y such that (x,y) is an edge, x!=u,v, y!=u,v.
        # And (u,x), (v,y) are not existing edges.
        
        # Let's pick a random edge (x,y) from the graph, excluding edges in short_cycle and edges incident to u or v.
        # To make it efficient, we can just pick random x, y.
        
        found_swap = False
        for _ in range(20): # Try a few random swaps
            if time.time() - start_time > timeout:
                return None
            x = random.randint(0, n-1)
            y = random.choice(adj[x])
            
            # Check conditions
            if x == u or x == v or y == u or y == v:
                continue
            if (min(x,y), max(x,y)) in short_cycle:
                continue
            
            # Try swap type 1: (u,v) & (x,y) -> (u,x) & (v,y)
            e1_new = (min(u,x), max(u,x))
            e2_new = (min(v,y), max(v,y))
            
            if e1_new in edge_set or e2_new in edge_set:
                # Try swap type 2: (u,v) & (x,y) -> (u,y) & (v,x)
                e1_new = (min(u,y), max(u,y))
                e2_new = (min(v,x), max(v,x))
                if e1_new in edge_set or e2_new in edge_set:
                    continue
                # Perform swap type 2
                # Remove (u,v) and (x,y)
                edge_set.remove((min(u,v), max(u,v)))
                edge_set.remove((min(x,y), max(x,y)))
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                # Add (u,y) and (v,x)
                edge_set.add(e1_new)
                edge_set.add(e2_new)
                adj[u].append(y)
                adj[y].append(u)
                adj[v].append(x)
                adj[x].append(v)
                found_swap = True
                break
            else:
                # Perform swap type 1
                edge_set.remove((min(u,v), max(u,v)))
                edge_set.remove((min(x,y), max(x,y)))
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                edge_set.add(e1_new)
                edge_set.add(e2_new)
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                found_swap = True
                break
        
        if not found_swap:
            # Couldn't find a valid swap for this edge, try another iteration
            continue

    return None

def construct(k, g):
    """
    Portfolio search:
    1. Try random regular graph construction with repair for n near Moore bound.
    2. Fallback to LUW algebraic construction.
    """
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Moore bound
    n_min = _moore_bound(k, g)
    
    # Ensure n * k is even
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    best = None
    best_n = float('inf')
    
    # Search n from n_min upwards
    # We expect to find a solution very close to n_min.
    # Step size 2 to maintain parity of n*k (if k is odd, n must be even).
    # If k is even, n can be anything.
    step = 2 if k % 2 != 0 else 1
    
    current_n = n_min
    
    while time.time() - start_time < deadline - 10:
        if current_n > 5000: # Safety limit for search range
            break
            
        # Try to generate and repair
        found = False
        # Try a few random seeds for this n
        for attempt in range(5):
            if time.time() - start_time > deadline - 5:
                break
                
            edges, adj = _generate_random_regular_graph(current_n, k, seed=random.randint(0, 10**9))
            if edges is None:
                continue
                
            # Check girth
            if _check_girth(current_n, adj, g):
                best = (current_n, edges)
                best_n = current_n
                found = True
                break # Found valid graph for this n, move to next n? No, this n is minimal so far.
                # Actually, we want the SMALLEST n. So if we find one for current_n, we can return it immediately?
                # Yes, because we are iterating n from Moore bound upwards.
                # So we can return immediately.
            else:
                # Try to repair
                repaired = _repair_girth(current_n, k, edges, adj, g, timeout=2.0)
                if repaired is not None:
                    rep_edges, rep_adj = repaired
                    if _check_girth(current_n, rep_adj, g):
                        best = (current_n, rep_edges)
                        best_n = current_n
                        found = True
                        break
                        
        if found:
            return best
            
        current_n += step
        
    if best is not None:
        return best
        
    # Fallback: LUW Construction
    # This is the same as the original code's fallback
    def _luw_graph(k, g, d_override=None, q_override=None):
        q = q_override if q_override is not None else k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        elif q == 2:
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: x * y
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q
        
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
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    try:
        return _luw_graph(k, g)
    except Exception:
        return _luw_graph(k, g, d_override=max(3, g-5))

# EVOLVE-BLOCK-END
