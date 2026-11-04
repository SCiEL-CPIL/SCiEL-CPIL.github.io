import time
import random
import math
from collections import deque

def construct(k, g):
    if k != 4:
        # Fallback for other k: use LUW
        return luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Target girth is g (10)
    # Try n from 320 upwards
    # The Moore bound for (4,10) is 2 * (1 + 4 + 4^2) = 2 * 21 = 42? No.
    # Moore bound for regular graph: n >= 2 * (k^(d/2) - 1) / (k-1) if d is even.
    # d=10, k=4. n >= 2 * (4^5 - 1) / 3 = 2 * 1023 / 3 = 682? 
    # Wait, the Moore bound formula for girth g=2d:
    # n >= 2 * sum_{i=0}^{d-1} (k-1)^i = 2 * ( (k-1)^d - 1 ) / (k-2)
    # For k=4, d=5: n >= 2 * (3^5 - 1) / 2 = 3^5 - 1 = 242.
    # So n >= 243.
    # The known best is 320.
    
    best_n = None
    best_edges = None
    
    # We will try to find a graph for n=320 first.
    # If we fail, we might try slightly larger n if we have time, but 320 is the target.
    # Actually, if we can't find 320, finding 321 is a good improvement over 1024.
    
    n_target = 320
    max_n_tried = 320
    
    # We can try a few random restarts for n=320
    # If that fails, try n=321, 322... up to some limit or time out.
    
    for n in range(320, 400):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph of size n
        # Use local search
        found = local_search_4_reg(n, g, deadline)
        if found:
            best_n, best_edges = found
            break # Found the smallest n, stop
            
    if best_n is None:
        # If we couldn't find a small one, fall back to LUW
        return luw_construct(4, g)
        
    return best_n, best_edges

def luw_construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
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
        
    n = len(order)
    return n, edges

def random_4_regular(n):
    # Generate a random 4-regular graph using perfect matching composition
    # 4-regular = union of two 2-factors (perfect matchings)
    # Or just random pairing of stubs.
    
    # Stubs: each vertex has 4 stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    # Pair them up
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u != v:
            if u > v:
                u, v = v, u
            edges.add((u, v))
            
    # Check if it's simple and 4-regular
    # If self-loop or multi-edge, reject and retry?
    # For large n, probability of multi-edge is low.
    # But we need to ensure no loops.
    
    # Let's use a more robust method: 
    # 1. Create list of vertices with 4 stubs each.
    # 2. Shuffle stubs.
    # 3. Pair consecutive stubs.
    # 4. If any pair is (u,u) or duplicate edge, discard and retry.
    
    for _ in range(100):
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.add((u, v))
            
        if valid:
            return list(edges)
            
    # Fallback: return empty or raise error? 
    # Should not happen for n >= 10.
    return []

def local_search_4_reg(n, g, deadline):
    # Start with random graph
    edges = random_4_regular(n)
    if not edges:
        return None
        
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check initial girth
    current_girth = get_girth(adj, g)
    
    if current_girth >= g:
        return n, edges
        
    # Local search: Edge swaps
    # Pick two random edges (u,v) and (x,y)
    # Swap to (u,x) and (v,y) or (u,y) and (v,x)
    # Ensure no loops or multi-edges
    
    # To speed up, we can precompute a list of edges
    edge_list = list(edges)
    
    # We need a fast girth check.
    # We'll use a BFS-based check that stops early if girth < g is found.
    
    # Optimization: Only check girth if the swap is "promising"?
    # Hard to know. We'll just check.
    
    # To make it faster, we can use a heuristic: 
    # If we break a short cycle, girth might increase.
    
    # Let's just do random swaps and check girth.
    
    # We need to be careful with time.
    # n=320, E=640.
    # BFS from each vertex is O(n+E).
    # Total check O(n*(n+E)) ~ 320 * 1000 = 320,000 ops.
    # In Python, 320k ops is ~0.1s.
    # So we can do ~1000 checks in 100s.
    
    # We'll do 1000 swaps.
    
    for step in range(1000):
        if time.time() > deadline:
            break
            
        # Pick two random edges
        e1_idx = random.randint(0, len(edge_list)-1)
        e2_idx = random.randint(0, len(edge_list)-1)
        if e1_idx == e2_idx:
            continue
            
        u, v = edge_list[e1_idx]
        x, y = edge_list[e2_idx]
        
        # Try swap 1: (u,x) and (v,y)
        # Check validity
        if u != x and v != y:
            # Check if (u,x) or (v,y) already exists
            # We need an adjacency set for fast lookup
            adj_set = set()
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        adj_set.add((i, j))
                        
            # This is too slow to rebuild adj_set every time.
            # We should maintain adj_set incrementally.
            
            # Let's rewrite the loop to maintain state.
            pass
            
    # The above loop is inefficient. Let's implement a proper stateful local search.
    
    return None

def get_girth(adj, g_limit):
    # Returns girth if < g_limit, else g_limit
    # BFS from each vertex
    n = len(adj)
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # But this is not necessarily the shortest cycle through u and v
                    # However, if dist[u] + dist[v] + 1 < g_limit, we found a short cycle.
                    # Note: This detects *some* cycle, not necessarily the girth.
                    # But if *any* cycle is < g_limit, we return < g_limit.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return cycle_len
                        
    return g_limit

def construct(k, g):
    if k != 4:
        return luw_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We will try to find a valid graph for n=320
    # Using a more efficient local search
    
    n_target = 320
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        edges = random_4_regular(n_target)
        if not edges:
            continue
            
        adj = [[] for _ in range(n_target)]
        adj_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            if u > v:
                u, v = v, u
            adj_set.add((u, v))
            
        current_girth = get_girth(adj, g)
        
        if current_girth >= g:
            return n_target, edges
            
        # Local search
        # We will perform 1000 swaps
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Pick two random edges
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
                
            u, v = e1
            x, y = e2
            
            # Try swap: (u,x) and (v,y)
            # Check if u==x or v==y (loop)
            if u == x or v == y:
                continue
                
            # Check if (u,x) or (v,y) in adj_set
            ux = (min(u,x), max(u,x))
            vy = (min(v,y), max(v,y))
            
            if ux in adj_set or vy in adj_set:
                continue
                
            # Perform swap
            # Remove (u,v) and (x,y)
            adj_set.remove(e1)
            adj_set.remove(e2)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add (u,x) and (v,y)
            adj_set.add(ux)
            adj_set.add(vy)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Update edges list
            edges.remove(e1)
            edges.remove(e2)
            edges.append(ux)
            edges.append(vy)
            
            # Check girth
            new_girth = get_girth(adj, g)
            
            if new_girth >= g:
                return n_target, edges
            elif new_girth > current_girth:
                current_girth = new_girth
            else:
                # Revert
                adj_set.remove(ux)
                adj_set.remove(vy)
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj_set.add(e1)
                adj_set.add(e2)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                
                edges.remove(ux)
                edges.remove(vy)
                edges.append(e1)
                edges.append(e2)

    # If we didn't find one for 320, try 330
    # ...
    # For now, fall back to LUW if we run out of time or fail
    return luw_construct(4, g)
