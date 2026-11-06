import random
import time

def construct():
    n = 16
    deadline = time.time() + 110  # 10 second margin
    
    def build_grotzsch():
        # Grötzsch graph on 0..10
        # Cycle 0-1-2-3-4-0
        # 5 connected to 0,1
        # 6 connected to 1,2
        # 7 connected to 2,3
        # 8 connected to 3,4
        # 9 connected to 4,0
        # 10 connected to 5,6,7,8,9
        # Wait, standard Mycielski construction:
        # Original vertices 0..4 (C5)
        # New vertices 5..9
        # 5' connected to 0, 1? No.
        # Mycielski(G): V' = V U {v' | v in V} U {w}
        # Edges: E(G) U {(u, v') | (u,v) in E(G)} U {(v', w) | v' in V'}
        
        # Let's use the explicit edges from the seed or standard definition.
        # Seed:
        # cycle = [(i, (i + 1) % 5) for i in range(5)]
        # edges = [list(e) for e in cycle]
        # for u, w in cycle:
        #     edges += [[u, 5 + w], [w, 5 + u]]
        # return edges + [[5 + u, 10] for u in range(5)]
        
        # This creates:
        # 0-1, 1-2, 2-3, 3-4, 4-0
        # 0-6, 1-5, 1-7, 2-6, 2-8, 3-7, 3-9, 4-8, 4-5
        # 5-10, 6-10, 7-10, 8-10, 9-10
        
        adj = [0] * n
        def add_edge(u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            
        # C5
        for i in range(5):
            add_edge(i, (i + 1) % 5)
            
        # Mycielski part
        # u in 0..4, v in 0..4
        # if (u,v) in E(C5), add (u, 5+v) and (v, 5+u)
        for i in range(5):
            u = i
            v = (i + 1) % 5
            add_edge(u, 5 + v)
            add_edge(v, 5 + u)
            
        # Connect 5..9 to 10
        for i in range(5):
            add_edge(5 + i, 10)
            
        return adj

    best_edges = 0
    best_adj = None

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def is_valid(adj):
        # Check triangle-free
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        # Check not 3-colorable
        # Since we start with Grötzsch and only add edges, it remains 4-chromatic.
        # But to be safe, we can check 3-colorability.
        # Backtracking 3-coloring check.
        colors = [-1] * n
        def backtrack(idx):
            if idx == n:
                return True
            used = 0
            for v in range(n):
                if (adj[idx] >> v) & 1:
                    if colors[v] != -1:
                        used |= 1 << colors[v]
            for c in range(3):
                if not (used >> c) & 1:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
            colors[idx] = -1
            return False
        
        # Optimization: order vertices by degree
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        # Remap colors array to order? Or just use original indices.
        # Simple backtracking on 16 vertices is fast enough if we prune.
        # Actually, we know it's 4-chromatic if it contains Grötzsch.
        # So we can skip the check if we are sure we started with Grötzsch.
        # But what if we modify the graph significantly?
        # We only ADD edges. So if G0 is 4-chromatic, G0+e is 4-chromatic.
        # So we don't need to check 3-colorability!
        return True

    def try_extend(adj):
        # Greedy edge addition
        # List of all possible edges not in adj
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    candidates.append((u, v))
        
        # Randomize order
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if adding (u,v) creates triangle
            if adj[u] & adj[v] == 0:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        
        return adj

    # Run multiple restarts
    start_time = time.time()
    iterations = 0
    while time.time() < deadline:
        adj = build_grotzsch()
        adj = try_extend(adj)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        iterations += 1
        # Maybe add some randomization to the extension process
        # E.g., sometimes skip an edge to allow different paths?
        # Or use simulated annealing?
        # Given time limit, simple random restarts might suffice.
        # Let's add a slight perturbation: start with a random subset of edges?
        # No, we must keep the Grötzsch core to ensure 4-chromaticity.
        # So we just randomize the order of adding edges.

    # Convert best_adj to list of edges
    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
