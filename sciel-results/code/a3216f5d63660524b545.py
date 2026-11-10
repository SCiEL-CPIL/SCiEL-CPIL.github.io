"""Combined program: LUW fallback with probabilistic rewiring for (4,10) cage."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For g=10 (even), we need d such that girth >= d+4 or d+5 depending on parity.
    # Standard LUW: girth >= d + 5 for odd d, d + 4 for even d.
    # We want girth >= g.
    # If g is even, d+4 >= g => d >= g-4.
    # If g is odd, d+5 >= g => d >= g-5.
    if g % 2 == 0:
        d = g - 4
        if d < 4: d = 4
        if d % 2 != 0: d += 1 # Ensure d is even? No, LUW works for any d >= 3? 
        # Actually, the rule construction in the seed code handles d generically.
        # The seed code used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # Let's stick to the seed logic which was validated.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    else:
        d = max(3, g - 5)

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

def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using the configuration model."""
    if n * k % 2 != 0:
        return None
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Pair up stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, retry pairing (simplified: just swap with a random other stub)
            # For small n, we can just restart if this happens often, or fix it.
            # A better way: use the "swap" method to fix, but for initialization, 
            # let's just try a few times.
            # Actually, standard configuration model can produce multi-edges and self-loops.
            # We need a simple graph.
            # Let's use a different approach: start with a cycle and random swaps.
            return None # Fallback to simpler init if this fails
        
        if v in adj[u]:
            # Multi-edge, retry
            return None
        
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    return adj, edges

def _init_graph_simple(n, k):
    """Initialize a k-regular graph by starting with a cycle and adding random chords/edges."""
    # Start with a cycle 0-1-...-n-1
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        u = i
        v = (i + 1) % n
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    
    # We have degree 2. We need degree k.
    # We need to add (k-2)*n/2 more edges.
    # Randomly pick pairs of vertices with degree < k and connect them.
    # This is a greedy approach.
    attempts = 0
    max_attempts = 100000
    while attempts < max_attempts:
        # Find vertices with degree < k
        candidates = [i for i in range(n) if len(adj[i]) < k]
        if not candidates:
            break
        # Pick two random distinct candidates
        if len(candidates) < 2:
            break
        u, v = random.sample(candidates, 2)
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        attempts += 1
        
    # Check if all degrees are k
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj, edges

def _get_girth(adj, n, max_girth):
    """Calculate the girth of the graph using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    if min_girth == float('inf'):
        return float('inf')
    return min_girth

def _check_swap_validity(u, v, x, y, adj, g):
    """Check if swapping (u,v) and (x,y) with (u,x) and (v,y) is valid and increases/maintains girth."""
    # 1. Check for loops and multi-edges in new edges
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
        
    # 2. Temporarily remove old edges
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    
    # 3. Check if new edges create short cycles
    # We need to check if there is a path from u to x of length < g-1
    # and from v to y of length < g-1 in the current graph (without the new edges)
    
    def bfs_shortest_path(src, dst, max_depth):
        if src == dst:
            return 0
        visited = {src: 0}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == dst:
                    return d + 1
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    q.append((neighbor, d + 1))
        return -1
    
    # We need a path of length L such that L + 1 < g => L < g - 1
    max_path_len = g - 2
    
    dist_ux = bfs_shortest_path(u, x, max_path_len)
    if dist_ux != -1:
        # Cycle formed: u -> ... -> x -> u
        # Length is dist_ux + 1
        # We reject if this cycle is shorter than g
        # Actually, we reject if it creates a cycle < g.
        # The new edge (u,x) closes the path.
        # So cycle length is dist_ux + 1.
        # If dist_ux + 1 < g, it's a bad cycle.
        adj[u].add(v)
        adj[v].add(u)
        adj[x].add(y)
        adj[y].add(x)
        return False
        
    dist_vy = bfs_shortest_path(v, y, max_path_len)
    if dist_vy != -1:
        adj[u].add(v)
        adj[v].add(u)
        adj[x].add(y)
        adj[y].add(x)
        return False
        
    # 4. If valid, add new edges
    adj[u].add(x)
    adj[x].add(u)
    adj[v].add(y)
    adj[y].add(v)
    
    return True

def _rewire_to_girth(n, k, g, deadline):
    """Try to rewire a random k-regular graph to have girth >= g."""
    adj, edges = _init_graph_simple(n, k)
    if adj is None:
        return None, None
        
    current_girth = _get_girth(adj, n, g)
    
    if current_girth >= g:
        return n, edges
        
    # Rewiring loop
    while time.time() < deadline:
        # Pick random edges to swap
        # We need to pick two non-adjacent edges (u,v) and (x,y)
        # such that u!=x, u!=y, v!=x, v!=y
        u, v = random.choice(edges)
        # Find another edge not incident to u or v
        valid_edges = []
        # Optimization: just pick random edges and check
        for _ in range(10): # Try a few random edges
            x, y = random.choice(edges)
            if x == u or x == v or y == u or y == v:
                continue
            if (u, x) in [(a,b) for a,b in edges] or (u, y) in [(a,b) for a,b in edges]: # Rough check, better to use adj
                continue
            if v in adj[x] or y in adj[u]: # Check if new edges would be multi-edges
                continue
            # Check if (u,x) or (v,y) already exist
            if x in adj[u] or y in adj[v]:
                continue
            
            # Perform swap check
            if _check_swap_validity(u, v, x, y, adj, g):
                # Update edges list
                # Remove (u,v) and (x,y)
                # Add (u,x) and (v,y)
                # This is O(E) to update list, but E is small (2n)
                edges[:] = [e for e in edges if e != (u,v) and e != (v,u) and e != (x,y) and e != (y,x)]
                edges.append((u, x))
                edges.append((v, y))
                
                # Check girth
                new_girth = _get_girth(adj, n, g)
                if new_girth >= g:
                    return n, edges
                # We accept the swap if it doesn't decrease girth? 
                # Or just accept if it's valid?
                # Usually we accept if it improves or maintains.
                # Let's accept if new_girth >= current_girth
                if new_girth < current_girth:
                    # Revert? 
                    # For simplicity, let's just keep it if it's valid, hoping to escape local minima.
                    # But if we degrade, we might never reach g.
                    # Better: only accept if new_girth >= current_girth.
                    # Revert
                    # Swap back
                    # adj is already updated. We need to revert adj and edges.
                    # This is getting complex. 
                    # Let's just accept valid swaps and rely on random walk.
                    pass
                current_girth = new_girth

    return None, None

def construct(k, g):
    deadline = time.time() + 110  # Leave 40s margin for safety and LUW fallback
    
    if k == 4 and g == 10:
        # Try to find a 320-vertex graph
        n_target = 320
        adj, edges = _rewire_to_girth(n_target, k, g, deadline)
        if adj is not None:
            return n_target, edges
        
    # Fallback to LUW
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
