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

def _generate_random_k_regular(n, k):
    """Generate a random k-regular simple graph using configuration model + 2-switches."""
    if k % 2 != 0 or (n * k) % 2 != 0:
        return None
    
    # Configuration model
    half_edges = n * k // 2
    vertices = list(range(n))
    half_edges_list = []
    for v in vertices:
        half_edges_list.extend([v] * k)
    
    random.shuffle(half_edges_list)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(0, len(half_edges_list), 2):
        u, v = half_edges_list[i], half_edges_list[i + 1]
        if u == v:
            continue
        if v in adj[u]:
            # Multiple edge, skip or handle
            continue
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if regular
    for i in range(n):
        if len(adj[i]) != k:
            # Try to fix with 2-switches
            # This is complex, so just return None for now
            # or try a simpler approach
            pass
    
    # For now, return the graph even if not perfectly regular? No, must be regular.
    # Let's use a simpler method: random permutation pairing with retries
    # Actually, let's use a different method:
    # Generate a random k-regular graph by ensuring each vertex has k neighbors.
    
    # Better: Use the "random k-regular graph" algorithm by Bollobas
    # Or just try multiple times
    attempts = 100
    for _ in range(attempts):
        half_edges_list = []
        for v in vertices:
            half_edges_list.extend([v] * k)
        random.shuffle(half_edges_list)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges_list), 2):
            u, v = half_edges_list[i], half_edges_list[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid:
            # Check degrees
            all_reg = True
            for i in range(n):
                if len(adj[i]) != k:
                    all_reg = False
                    break
            if all_reg:
                return adj
    
    return None

def _girth_check(adj, n, g):
    """Check if graph has girth >= g using non-backtracking walks."""
    # For each vertex, do a BFS that doesn't backtrack
    # If we find a cycle of length < g, return False
    
    # This is expensive for n=1000+.
    # Simpler: BFS from each vertex, but track path to avoid backtracking.
    # Actually, standard BFS finds shortest path, which corresponds to shortest cycle through that edge?
    # No, girth is min cycle length.
    # We can check for cycles of length < g by checking if there's a path of length < g between adjacent vertices?
    # No.
    
    # Efficient check: For each edge (u,v), remove it and check if dist(u,v) < g-1.
    # If yes, cycle of length < g.
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    for u, v in edges:
        # BFS from u to v, not using edge (u,v)
        # If dist < g-1, then cycle length = dist + 1 < g
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= g - 1:
                continue
            for neighbor in adj[node]:
                if neighbor == v and node == u:
                    continue # Skip the edge (u,v)
                if neighbor == u and node == v:
                    continue # Skip the edge (v,u)
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                    if neighbor == v:
                        return False # Found path of length dist+1 < g-1? 
                        # Wait, if dist+1 < g-1, then cycle < g.
                        # We stopped if dist >= g-1. So if we reach v, dist < g-1.
                        # So cycle length = dist + 1 < g.
    
    return True

def _switch_improves_girth(adj, n, u, v, w, x, g):
    """
    Check if switching (u,v)-(w,x) to (u,w)-(v,x) improves or maintains girth >= g.
    This is a local check.
    """
    # This is complex to do exactly.
    # Approximation: Check if the new edges create short cycles locally.
    # If u-w exists, invalid. If v-x exists, invalid.
    if w in adj[u] or x in adj[v]:
        return False
    
    # Check if adding (u,w) creates a short cycle
    # Check if adding (v,x) creates a short cycle
    # And removing (u,v) and (w,x) doesn't create issues (it can't, as it removes edges)
    
    # We need to ensure no cycle of length < g is created.
    # The only new cycles involve the new edges.
    # So we check if dist(u,w) < g-1 in the graph without (u,w) (which is current graph).
    # And dist(v,x) < g-1.
    
    def has_path(a, b, max_len, exclude_edge=None):
        if a == b:
            return 0
        visited = {a: 0}
        queue = [(a, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= max_len:
                continue
            for neighbor in adj[node]:
                if exclude_edge and ((node, neighbor) == exclude_edge or (neighbor, node) == exclude_edge):
                    continue
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                    if neighbor == b:
                        return dist + 1
        return None
    
    # Check u-w
    # The edge (u,w) is not in the graph yet.
    # So we check if there is a path between u and w of length < g-1.
    # If yes, adding (u,w) creates a cycle of length < g.
    # Note: The path cannot use (u,w) as it doesn't exist.
    
    # But wait, the switch also removes (u,v) and (w,x).
    # So the path for u-w should not use (u,v) or (w,x)?
    # Actually, the path is in the graph AFTER the switch.
    # So we should check in the graph with (u,v) and (w,x) removed and (u,w) and (v,x) added.
    # This is getting complicated.
    
    # Simplified heuristic:
    # 1. Check if u and w are already close (dist < g-1) in current graph.
    # 2. Check if v and x are already close (dist < g-1) in current graph.
    # 3. Also check if the switch creates a 4-cycle: u-w-x-v-u?
    #    If (w,x) is removed, and (u,w) added, (v,x) added.
    #    If (u,v) was removed.
    #    Cycle: u-w-x-v-? No, (u,v) is gone.
    #    Cycle: u-w-x-v-u? No, (u,v) is gone.
    #    So no 4-cycle from this specific switch unless there are other edges.
    
    # Let's just do a local BFS check for the new edges in the current graph (approximation).
    
    # If this is too slow or inaccurate, we might fail.
    # Given time constraints, let's try a simpler approach:
    # Just try random switches and check the full girth periodically?
    # No, that's too slow.
    
    # Let's try a different strategy:
    # Start with a random graph.
    # While time remains:
    #   Pick a random edge (u,v).
    #   Pick a random other edge (w,x) not sharing vertices.
    #   Try switch.
    #   Check if new graph has girth >= g (full check).
    #   If yes, accept.
    #   If no, reject.
    
    # But full check is O(n^2) or worse.
    
    # Alternative:
    # Use the fact that we want to INCREASE girth.
    # So we start with a graph that might have small cycles.
    # We want to break them.
    # A switch breaks cycles that use the removed edges and creates cycles that use the new edges.
    # So we should prefer switches that break existing short cycles.
    
    # Let's identify short cycles in the current graph.
    # Then try to break them.
    
    return True # Placeholder, needs implementation

def _construct_via_switching(k, g, n, time_limit=10.0):
    start_time = time.time()
    
    # 1. Generate initial random k-regular graph
    adj = _generate_random_k_regular(n, k)
    if adj is None:
        return None
    
    # 2. Check initial girth
    # If already >= g, return
    if _girth_check(adj, n, g):
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        return n, edges
    
    # 3. Switching loop
    # We need to improve girth.
    # Strategy: Find a short cycle, pick two edges in it, and try to switch them with random other edges to break the cycle.
    
    # For now, let's just do random switches and check girth occasionally.
    # This is inefficient but might work for small improvements.
    
    # Better:
    # While time < limit:
    #   Find a cycle of length < g.
    #   If none, return success.
    #   Pick two edges in this cycle.
    #   Try random switches involving these edges.
    #   If a switch breaks the cycle and doesn't create a shorter one, accept.
    
    # This is complex to implement correctly.
    # Given the 120s limit, let's try a simpler heuristic:
    # Just try random switches for a while, and check girth every 1000 switches.
    
    num_switches = 0
    last_check = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random edge (u,v)
        u = random.randint(0, n - 1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # Pick random edge (w,x) not sharing u or v
        w = random.randint(0, n - 1)
        if w in (u, v) or not adj[w]:
            continue
        x = random.choice(list(adj[w]))
        if x in (u, v) or x == w:
            continue
        if (u, x) in [(a, b) for a in adj[u] for b in adj[a]]: # Check if u-x exists?
            # This is O(k^2)
            pass
        
        # Check if u-x and v-w exist
        if x in adj[u] or w in adj[v]:
            continue
        
        # Perform switch
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(w)
        adj[w].add(v)
        
        num_switches += 1
        
        # Check girth periodically
        if num_switches % 100 == 0:
            if _girth_check(adj, n, g):
                edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                return n, edges
    
    # Final check
    if _girth_check(adj, n, g):
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try switching
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1120, 1150, 1200, 1500]
        
        # Time budget: 100s for switching, 10s for fallback
        time_per_attempt = 10.0 
        
        for n in target_ns:
            res = _construct_via_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
