import random
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    # (Include the original code as a nested function or inline)
    
    # Heuristic Search
    # Target: 3-regular, girth >= g
    # Start n from known cage size or Moore bound
    # Known cage for (3,16) is 936.
    
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Try to find a graph with n starting from 936
    # If we find one, return it.
    # If not, increase n.
    
    # For this specific task, we know 936 works.
    # We will try to find a valid graph for n=936, 940, 950, etc.
    
    def generate_random_3_reg(n):
        # Generate a random 3-regular graph using configuration model
        # Half-edges: 3n stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        for i in range(0, 3*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u != v:
                edges.append((u, v))
            else:
                # Loop, retry or handle
                # Simple retry
                # For now, just skip and hope, or regenerate
                # Better: use a more robust generator
                pass
        # Check for multi-edges
        edge_set = set()
        valid_edges = []
        for u, v in edges:
            if u > v: u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                valid_edges.append((u, v))
            else:
                # Multi-edge, invalid for simple graph
                # We need a different generator or retry
                pass
        if len(valid_edges) != n * 3 // 2:
            return None
        return valid_edges

    def get_adjacency(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def check_girth(n, adj, min_girth):
        # BFS from each vertex up to depth min_girth//2
        # If we find a back-edge to a visited node at depth < min_girth//2, girth is smaller
        # Actually, girth is the length of the shortest cycle.
        # BFS from each node, if we find a neighbor already visited at a smaller depth, we have a cycle.
        # Cycle length = depth[u] + depth[v] + 1
        # We need min cycle length >= min_girth
        
        for start in range(n):
            depth = [-1] * n
            parent = [-1] * n
            q = deque()
            q.append(start)
            depth[start] = 0
            while q:
                u = q.popleft()
                if depth[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = depth[u] + depth[v] + 1
                            if cycle_len < min_girth:
                                return False, cycle_len
        return True, min_girth

    def improve_girth(n, edges, adj, target_girth, max_iters=10000):
        # Local search: find short cycle, break it with a 2-switch
        # 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x)
        # Must ensure no multi-edges or loops
        
        for _ in range(max_iters):
            is_valid, g = check_girth(n, adj, target_girth)
            if is_valid:
                return edges, adj, True
            
            # Find a short cycle
            # We can modify check_girth to return the cycle
            # For simplicity, let's just do random 2-switches and hope
            # Or better, find the specific cycle
            
            # Pick a random edge to remove
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Pick another random edge
            # Ensure it doesn't share vertices with u,v to keep it simple (not strictly necessary for 2-switch, but avoids loops)
            # Actually, 2-switch can involve edges that don't share vertices.
            # Let's pick a random other edge (w, x)
            # To avoid loops, we need u!=w, u!=x, v!=w, v!=x
            # And to avoid multi-edges, (u,w) and (v,x) must not exist
            
            # This is getting complex. Let's try a simpler "random restart" approach.
            # Generate new random graph.
            # If we have time, we can try multiple random graphs.
            
            # Given the time limit, and that 936 is small, 
            # we can just try generating random graphs until we find one with girth >= 16.
            # Probability might be low, but with 100s of seconds, we can try thousands.
            
            # Let's optimize the check_girth to be faster.
            pass
        
        return edges, adj, False

    # Strategy:
    # 1. Try LW construction (fast, guaranteed valid, but large n).
    # 2. Try to find a smaller graph (n ~ 936) via random search.
    # 3. If found, return it. Else return LW.
    
    # Implement LW construction first to have a valid baseline
    # (Copy from original)
    
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
    index, order, edges_lw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_lw.append((index[vertex], index[other]))
    n_lw = len(order)
    
    # Now try to find a smaller graph
    # Start n from 936
    # If k != 3 or g != 16, this heuristic might not work well.
    # The task specifies k and g are passed.
    # If k=3, g=16, we try this.
    # If other parameters, we fall back to LW or a generic search.
    
    if k == 3 and g == 16:
        # Try to find a graph with n < n_lw
        # We know n=936 works.
        # Let's try n=936, 940, 950...
        # We will use a randomized greedy construction.
        
        def try_find_graph(n_target):
            # Randomly generate 3-regular graphs and check girth
            # To make it faster, we can use a specific construction like Circulant
            # or just random restarts.
            
            # Given the low probability of a random graph having girth 16,
            # we might need a smarter heuristic.
            # However, for the purpose of beating LW (354k), even n=1000 is a win.
            # If we can't find it quickly, we return LW.
            
            # Let's try a few random graphs.
            for _ in range(100): # 100 attempts
                # Generate random 3-regular graph
                # Using configuration model with rejection of loops/multi-edges
                edges = []
                while len(edges) < n_target * 3 // 2:
                    u = random.randint(0, n_target - 1)
                    v = random.randint(0, n_target - 1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if (u, v) in set(map(tuple, map(sorted, edges))): continue
                    edges.append((u, v))
                
                # Check degree
                deg = [0] * n_target
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue
                
                # Check girth
                adj = [[] for _ in range(n_target)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Fast girth check
                min_g = n_target + 1
                for s in range(n_target):
                    depth = [-1] * n_target
                    q = deque([s])
                    depth[s] = 0
                    while q:
                        u = q.popleft()
                        if depth[u] >= min_g // 2:
                            continue
                        for v in adj[u]:
                            if depth[v] == -1:
                                depth[v] = depth[u] + 1
                                q.append(v)
                            else:
                                if v != (adj[u][0] if len(adj[u])==1 else -1): # Simplified parent check
                                    # Proper parent check needed
                                    pass
                # This BFS is a bit messy. Let's use a standard one.
                
                # Re-implement girth check cleanly
                girth = n_target + 1
                for s in range(n_target):
                    dist = [-1] * n_target
                    par = [-1] * n_target
                    q = deque([s])
                    dist[s] = 0
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                par[v] = u
                                q.append(v)
                            else:
                                if v != par[u]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < girth:
                                        girth = cycle_len
                    if girth < g:
                        break
                
                if girth >= g:
                    return n_target, edges
            
            return None
        
        # Try a few n values
        for n_try in range(936, 1200, 10):
            if time.time() > deadline:
                break
            res = try_find_graph(n_try)
            if res:
                return res
        
    return n_lw, edges_lw
