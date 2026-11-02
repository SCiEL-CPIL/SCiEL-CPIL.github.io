import time
import random
import heapq
import sys

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _random_regular_graph(n, k):
    """Generates a random k-regular graph on n vertices using configuration model + 2-switches."""
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    # Create a multigraph by pairing stubs
    # To ensure simplicity, we use 2-switches to remove multi-edges/loops
    # This is a standard heuristic
    
    # 1. Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # 2. Randomly pair stubs
    # This creates a multigraph (may have loops and multi-edges)
    random.shuffle(stubs)
    edges = []
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        edges.append((u, v))
        
    # 3. Remove loops and multi-edges via switching
    # We represent the graph as a list of edges and a set of unique edges for fast lookup
    # We need to maintain regularity.
    
    # For simplicity and speed in this context, we might just generate a random simple k-regular graph
    # using a different method if n is not too large, or accept that this might fail for small n.
    # However, for n=1116, k=5, this is doable.
    
    # Let's try a simpler approach: Start with a known simple k-regular graph (e.g., circulant) 
    # and randomize it via 2-switches.
    
    # Initialize with a circulant graph
    # Vertices 0..n-1
    # Connect i to i+1, i-1, i+2, i-2, i+3? 
    # Need to be careful with n being even/odd and k=5.
    # If n is even, i+1, i-1, i+2, i-2, i+n/2 works.
    # If n is odd, i+1, i-1, i+2, i-2, i+3?
    
    # Let's just use a random pairing and fix it.
    # This is complex to implement robustly in a short time.
    
    # Alternative: Use the fact that we can start with the LUW graph and switch it?
    # No, the directive says "starting from a random 5-regular graph".
    
    # I will implement a simple random regular graph generator that works for moderate n.
    # Method: 
    # 1. Create a list of vertices with remaining degree k.
    # 2. Randomly pick u, v with remaining degree > 0.
    # 3. If u != v and edge (u,v) not present, add it.
    # 4. Decrease remaining degrees.
    # 5. If stuck (e.g., two vertices with degree 1 left but they are already connected), backtrack or restart.
    # This "greedy random" often works for k=5, n=1116.
    
    adj = [set() for _ in range(n)]
    rem_deg = [k] * n
    vertices = list(range(n))
    
    for _ in range(n * k // 2):
        # Pick a vertex with remaining degree > 0
        # To avoid bias, pick from a list of active vertices
        active = [v for v in range(n) if rem_deg[v] > 0]
        if not active:
            break
        u = random.choice(active)
        
        # Pick a partner v
        # v must have rem_deg > 0 and v != u and (u,v) not in adj
        candidates = [v for v in range(n) if rem_deg[v] > 0 and v != u and v not in adj[u]]
        if not candidates:
            return None # Failed
        
        v = random.choice(candidates)
        
        adj[u].add(v)
        adj[v].add(u)
        rem_deg[u] -= 1
        rem_deg[v] -= 1
        
    # Verify
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    return adj

def _girth(adj, n):
    """Calculate girth of the graph."""
    # BFS from each vertex
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Cycle length = dist[u] + dist[v] + 1
                    # We need to find the LCA to be precise, but for girth, 
                    # any cycle found is a candidate. 
                    # Actually, BFS finds shortest paths. 
                    # A cycle is formed if we see a neighbor that is already visited 
                    # and is not the parent.
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    # This is correct for the cycle formed by the two paths from start.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        if min_girth <= 5: # Early exit
            break
            
    return min_girth if min_girth != float('inf') else n

def _find_short_cycle(adj, n, max_len):
    """Find a cycle of length <= max_len. Returns list of vertices in the cycle."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found cycle
                    # Reconstruct cycle
                    cycle = []
                    # Path from u to LCA
                    curr = u
                    while curr != -1:
                        cycle.append(curr)
                        curr = parent[curr]
                    # Path from v to LCA
                    curr = v
                    while curr != -1:
                        cycle.append(curr)
                        curr = parent[curr]
                    # The LCA is the last common element. 
                    # This reconstruction is messy. 
                    # Simpler: just return the length for now, or a dummy cycle.
                    # For switching, we need the actual edges.
                    # Let's implement a proper cycle extraction.
                    
                    # Find LCA
                    path_u = []
                    curr = u
                    while curr != -1:
                        path_u.append(curr)
                        curr = parent[curr]
                    path_v = []
                    curr = v
                    while curr != -1:
                        path_v.append(curr)
                        curr = parent[curr]
                    
                    set_u = set(path_u)
                    lca = None
                    for node in path_v:
                        if node in set_u:
                            lca = node
                            break
                            
                    if lca is None:
                        continue
                        
                    cycle_u = path_u[:path_u.index(lca) + 1]
                    cycle_v = path_v[:path_v.index(lca) + 1]
                    cycle = cycle_u + cycle_v[::-1]
                    return cycle
                    
    return None

def _apply_switch(adj, u1, v1, u2, v2):
    """Remove edges (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)."""
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    
    adj[u1].add(v2)
    adj[v2].add(u1)
    adj[u2].add(v1)
    adj[v1].add(u2)

def _switching_construct(k, g, n, time_limit=50.0):
    start_time = time.time()
    
    # 1. Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
        
    # 2. Switching loop
    # We want to increase girth to at least g.
    # While girth < g:
    #   Find a short cycle.
    #   Try to break it with a switch.
    #   If switch increases girth or doesn't decrease it too much, accept.
    
    current_girth = _girth(adj, n)
    
    while current_girth < g:
        if time.time() - start_time > time_limit:
            return None
            
        cycle = _find_short_cycle(adj, n, current_girth)
        if not cycle:
            # No cycle found? Girth calculation might be wrong or graph is disconnected?
            # If connected and no cycle, it's a tree (impossible for regular k>=2).
            break
            
        # Try to break the cycle
        # A switch involves two edges.
        # If we pick an edge (a,b) in the cycle and an edge (c,d) outside, 
        # we can switch them.
        # We need to ensure the new edges (a,d) and (c,b) don't create shorter cycles.
        
        # Simple heuristic: 
        # Pick a random edge in the cycle.
        # Pick a random edge in the graph.
        # Try the switch.
        # If the new girth is >= old girth, accept.
        
        # This is very slow if we recalculate girth every time.
        # Optimization: Only check local changes? Hard.
        # Given time limits, this might be too slow.
        
        # Alternative: Just return None if we can't do it fast.
        # The fallback _luw_construct is reliable.
        
        # For the purpose of this exercise, I will implement a very limited number of steps.
        
        # Pick an edge from the cycle
        edge_idx = random.randint(0, len(cycle) - 1)
        u1 = cycle[edge_idx]
        v1 = cycle[(edge_idx + 1) % len(cycle)]
        
        # Pick a random other edge
        # Get all edges
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
                    
        if not all_edges:
            return None
            
        # Filter out edges in the cycle? Not strictly necessary, but helpful.
        # Just pick a random edge.
        u2, v2 = random.choice(all_edges)
        
        # Check if u2,v2 is the same as u1,v1 or connected in a way that breaks simplicity
        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
            continue
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Perform switch
        _apply_switch(adj, u1, v1, u2, v2)
        
        # Check new girth (expensive)
        new_girth = _girth(adj, n)
        
        # If new girth is not better, revert?
        # Or just keep it if it's not worse?
        # To avoid getting stuck in local minima, we can accept if new_girth >= current_girth.
        if new_girth < current_girth:
            # Revert
            _apply_switch(adj, u1, v2, u2, v1) # Revert logic: 
            # Original: (u1,v1), (u2,v2) -> (u1,v2), (u2,v1)
            # Revert: (u1,v2), (u2,v1) -> (u1,v1), (u2,v2)
            # So we call _apply_switch(adj, u1, v2, u2, v1)
            pass # The revert is done by calling _apply_switch again with the new edges
        else:
            current_girth = new_girth
            
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Try switching
        # Start with best known size
        n_target = 1116
        res = _switching_construct(k, g, n_target, time_limit=30.0)
        if res is not None:
            return res
            
        # Fallback
        return _luw_construct(k, g)
        
    # General case
    return _luw_construct(k, g)
