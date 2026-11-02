import time
import random
from collections import deque

def _luw_construct(k, g):
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

def _compute_girth_fast(adj, n, max_girth=15):
    """Compute girth using BFS from each vertex with early termination."""
    min_girth = max_girth + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= max_girth:
                            return min_girth
    return min_girth

def _generate_k_regular(n, k):
    """Generate a simple k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj

def _try_switch(adj, n, u1, v1, u2, v2, current_girth, target_girth):
    """Try a 2-switch: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2).
    Returns True if switch is valid and doesn't decrease girth below target."""
    # Check for conflicts
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        return False

    # Apply switch
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    adj[u1].add(u2)
    adj[u2].add(u1)
    adj[v1].add(v2)
    adj[v2].add(v1)

    # Check girth locally - only need to check cycles involving the new edges
    # A cycle using (u1,u2) must go from u1 to u2 without using that edge
    # Similarly for (v1,v2)
    # For efficiency, do a limited BFS
    new_girth = _compute_girth_fast(adj, n, max_girth=target_girth - 1)
    
    # Revert
    adj[u1].remove(u2)
    adj[u2].remove(u1)
    adj[v1].remove(v2)
    adj[v2].remove(v1)
    adj[u1].add(v1)
    adj[v1].add(u1)
    adj[u2].add(v2)
    adj[v2].add(u2)

    if new_girth >= target_girth:
        # Re-apply the switch
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        return True
    return False

def _edge_switching(adj, n, k, target_girth, time_budget):
    """Improve girth using 2-switches."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Compute current girth
    current_girth = _compute_girth_fast(adj, n, max_girth=target_girth)
    if current_girth >= target_girth:
        return True

    # Build edge list
    def get_edges():
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        edges = get_edges()
        if len(edges) < 2:
            break
        
        # Pick two random edges
        i1 = random.randint(0, len(edges) - 1)
        i2 = random.randint(0, len(edges) - 1)
        if i1 == i2:
            continue
        
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]
        
        # Ensure disjoint
        if len({u1, v1, u2, v2}) < 4:
            continue
        
        if _try_switch(adj, n, u1, v1, u2, v2, current_girth, target_girth):
            current_girth = _compute_girth_fast(adj, n, max_girth=target_girth)
            if current_girth >= target_girth:
                return True
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    if k == 5 and g == 9:
        # Try multiple n values, starting from smaller
        target_ns = [850, 900, 950, 1000, 1050, 1100]
        
        for n in target_ns:
            if time.time() > deadline - 10:
                break
            
            # Time budget for this n
            time_budget = min(20.0, deadline - time.time() - 5)
            if time_budget < 5:
                break
            
            # Try multiple random instances
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                
                adj = _generate_k_regular(n, k)
                if adj is None:
                    continue
                
                # Check if already good
                girth = _compute_girth_fast(adj, n, max_girth=g)
                if girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                
                # Try edge switching
                remaining = deadline - time.time() - 5
                if remaining < 5:
                    break
                
                if _edge_switching(adj, n, k, g, remaining):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g)
