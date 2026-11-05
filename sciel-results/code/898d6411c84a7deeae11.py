"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110  # Safety margin
    
    if k == 4 and g == 10:
        # Attempt to find the Foster Cage (320 vertices) via local search
        result = _search_cage_4_10(deadline)
        if result is not None:
            return result
        # Fallback to LUW construction
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)

def _luw_graph(k, g):
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _search_cage_4_10(deadline):
    n = 320
    # Try a few random restarts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        adj = _random_4reg(n)
        if adj is None:
            continue
        
        # Local search
        adj = _local_search_girth(adj, n, 10, deadline)
        
        # Verify
        if _verify_girth(adj, n, 10):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        # If time is running out, break
        if time.time() > deadline - 10:
            break
            
    return None

def _random_4reg(n):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    if not valid:
        return None
    return adj

def _calc_girth(adj, n, max_depth=4):
    # Returns shortest cycle length, or infinity if none found up to max_depth*2
    # For girth 10, we need to check cycles of length 3..9.
    # BFS from each vertex up to depth 4.
    for start in range(n):
        visited = {start: 0}
        parent = {start: -1}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == parent[node]:
                    continue
                if nb in visited:
                    # Cycle found
                    # Length is depth + 1 + visited[nb]? 
                    # Actually, if we just want to know if girth < 10, 
                    # finding ANY cycle of length < 10 is enough.
                    # The length of the cycle formed by node->nb is depth + 1 + visited[nb]?
                    # No, visited[nb] is the depth from start.
                    # The cycle is start -> ... -> node -> nb -> ... -> start.
                    # Length = depth + 1 + visited[nb].
                    # We need to ensure this length is < 10.
                    cycle_len = depth + 1 + visited[nb]
                    if cycle_len < 10:
                        return cycle_len
                else:
                    visited[nb] = depth + 1
                    parent[nb] = node
                    queue.append((nb, depth + 1))
    return 10 # Assume girth >= 10 if no short cycle found

def _verify_girth(adj, n, g):
    # Strict verification
    for start in range(n):
        # Non-backtracking BFS
        # State: (node, prev)
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start))
            visited[(nb, start)] = 1
            
        while queue:
            node, prev = queue.popleft()
            depth = visited[(node, prev)]
            if depth >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    if depth + 1 < g:
                        return False
                if (nb, node) not in visited:
                    visited[(nb, node)] = depth + 1
                    queue.append((nb, node))
    return True

def _local_search_girth(adj, n, target_girth, deadline):
    current_girth = _calc_girth(adj, n)
    if current_girth >= target_girth:
        return adj
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find a short cycle
        cycle_len = _calc_girth(adj, n)
        if cycle_len >= target_girth:
            break
            
        # Try random 2-switches
        # Get list of edges
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    
        random.shuffle(edges)
        # Try a limited number of switches
        for _ in range(100):
            if time.time() > deadline:
                return adj
            if len(edges) < 2:
                break
            i, j = random.sample(range(len(edges)), 2)
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # Check for multi-edges in new config
            # Remove (u1,v1) and (u2,v2)
            # Add (u1,u2) and (v1,v2)
            # Check if u1==u2 or v1==v2 (self loop)
            if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
                continue
            # Check if (u1,u2) or (v1,v2) already exist
            if u2 in adj[u1] or v2 in adj[v1]:
                continue
            if u1 in adj[u2] or v1 in adj[v2]:
                continue
                
            # Perform switch
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            # Update edges list? Or just re-calc girth
            new_girth = _calc_girth(adj, n)
            if new_girth > current_girth:
                current_girth = new_girth
                improved = True
                break
            else:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
    return adj
# EVOLVE-BLOCK-END
