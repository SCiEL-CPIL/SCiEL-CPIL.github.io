import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100
    
    # We aim for n close to the known best 1116, but even.
    # Let's start with n = 1120.
    n = 1120
    
    # 1. Generate random k-regular graph
    def gen_random_reg(n, k):
        stubs = [i for i in range(n) for _ in range(k)]
        while True:
            random.shuffle(stubs)
            edges = []
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                edges.append((u, v))
            if not valid:
                continue
            
            # Check simple
            edge_set = set()
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if valid:
                return edges
        return None

    # 2. Build adjacency and check girth
    def get_short_cycles(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        short_cycles = []
        # Check for cycles of length 3, 4, 5, 6, 7, 8
        # For efficiency, we just need to know if ANY exist.
        # We can do BFS from each vertex up to depth (g-1)//2.
        # If we find a back-edge forming a cycle < g, record it.
        
        for u in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            queue = [(u, 0)]
            dist[u] = 0
            while queue:
                curr, d = queue.pop(0)
                if d >= (g-1)//2:
                    continue
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = d + 1
                        parent[nb] = curr
                        queue.append((nb, d + 1))
                    else:
                        if nb != parent[curr]:
                            # Potential cycle
                            # The cycle length is dist[curr] + dist[nb] + 1
                            # This is an upper bound on the shortest cycle through this edge?
                            # Actually, if dist[curr] + dist[nb] + 1 < g, we have a short cycle.
                            if dist[curr] + dist[nb] + 1 < g:
                                # We found a short cycle.
                                # To break it, we need to swap an edge in the cycle.
                                # Let's just return the edge (curr, nb) as a "problem edge".
                                return [(curr, nb)] # Return just one to start
        return []

    edges = gen_random_reg(n, k)
    if edges is None:
        return 0, []

    # 3. Local search
    # We will try to remove short cycles by swapping edges.
    # We need to be careful not to create new ones.
    
    # To make this tractable, we will limit the number of swaps.
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Check for short cycles
        problem_edges = get_short_cycles(n, edges, g)
        
        if not problem_edges:
            # Verify girth is >= g
            # The BFS check above is sufficient for "no cycle < g"
            return n, edges
        
        # Pick a random problem edge (curr, nb)
        # We need to find an edge to swap with.
        # A 2-switch: (u, v) and (x, y) -> (u, x) and (v, y)
        # We pick one of the problem edges, say (u, v).
        # We pick a random other edge (x, y).
        # We swap.
        # We must ensure we don't create loops or multi-edges.
        
        u, v = problem_edges[0]
        
        # Find index of (u, v) in edges
        idx1 = -1
        for i, (a, b) in enumerate(edges):
            if (a == u and b == v) or (a == v and b == u):
                idx1 = i
                break
        
        if idx1 == -1:
            continue
            
        # Pick random idx2
        idx2 = random.randint(0, len(edges) - 1)
        if idx2 == idx1:
            continue
            
        x, y = edges[idx2]
        
        # Try swap (u, x) and (v, y)
        # Check validity:
        # 1. u != x, v != y (no loops)
        # 2. (u, x) not in edges (except the ones we are removing)
        # 3. (v, y) not in edges
        
        # To check "not in edges", we can use a set.
        # For speed, we might skip this check and rely on the next iteration's check?
        # No, we must maintain simple graph.
        
        # Create set of edges
        edge_set = set((min(a, b), max(a, b)) for a, b in edges)
        
        # Remove the two edges
        e1 = (min(u, v), max(u, v))
        e2 = (min(x, y), max(x, y))
        edge_set.remove(e1)
        edge_set.remove(e2)
        
        # Check new edges
        new_e1 = (u, x)
        new_e2 = (v, y)
        
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            # Loop
            edge_set.add(e1)
            edge_set.add(e2)
            continue
            
        key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
        key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
        
        if key1 in edge_set or key2 in edge_set:
            # Multi-edge
            edge_set.add(e1)
            edge_set.add(e2)
            continue
            
        # Perform swap
        edges[idx1] = new_e1
        edges[idx2] = new_e2
        edge_set.add(key1)
        edge_set.add(key2)
        
        # Rebuild edges list from set? No, we updated list.
        # But we need to keep edge_set in sync for next iteration?
        # The next iteration rebuilds it.

    # If we fail to find a girth-9 graph, return the best effort?
    # No, must be valid.
    # If we run out of time, we return the current graph.
    # But it might have short cycles.
    # We MUST return a valid graph.
    # If the local search fails, we should probably return a known valid graph.
    # Since we don't have one, we might return 0, [] (score 0).
    # To avoid score 0, we should ensure the graph IS valid.
    # If the loop finishes, we check girth.
    # If girth < g, we cannot return it.
    
    # Let's add a final check.
    final_problem = get_short_cycles(n, edges, g)
    if not final_problem:
        return n, edges
    
    # If we still have short cycles, we failed.
    # Return 0, []
    return 0, []
